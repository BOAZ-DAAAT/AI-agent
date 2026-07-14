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
    route_kind_raw = str(parsed.get("route_kind") or "").strip().lower()
    route_kind = route_kind_raw
    if route_kind not in {"simple", "comprehensive"}:
        raise ValueError(f"unsupported route_kind: {route_kind or 'empty'}")

    task_type = "data_mart_build" if route_kind == "comprehensive" else "query_answer"
    question_type = str(parsed.get("question_type") or ("mart_build" if route_kind == "comprehensive" else "detail"))
    requested_output = "create_table" if route_kind == "comprehensive" else "execute_and_answer"
    selected_tables = _list_of_str(parsed.get("selected_join_tables"))
    relevant_tables = _list_of_str(parsed.get("relevant_tables")) or list(selected_tables)
    candidate_tables = _list_of_str(parsed.get("candidate_tables")) or list(dict.fromkeys(selected_tables + relevant_tables))
    validation_contract = dict(parsed.get("validation_contract") or {})
    expected_result_shape = "datamart_creation" if route_kind == "comprehensive" else "table_preview"
    required_columns = _list_of_str(parsed.get("required_columns"))
    required_aggregations = _list_of_str(parsed.get("required_aggregations"))
    dimensions = _list_of_str(parsed.get("dimensions"))
    filters = _list_of_str(parsed.get("filters"))
    target_metric = str(parsed.get("target_metric") or "")

    validation_contract["expected_result_shape"] = expected_result_shape
    validation_contract.setdefault("required_columns", list(required_columns))
    validation_contract.setdefault("required_aggregations", list(required_aggregations))
    validation_contract.setdefault("required_tables", list(selected_tables or relevant_tables))
    validation_contract.setdefault("expected_aliases", [])
    validation_contract.setdefault("target_metric", target_metric)
    validation_contract.setdefault("dimensions", list(dimensions))
    validation_contract.setdefault("target_table", None)
    validation_contract.setdefault("mart_policy", "prefer_row_preserving" if route_kind == "comprehensive" else None)

    has_core_signal = bool(
        parsed.get("route_kind")
        or selected_tables
        or relevant_tables
        or candidate_tables
        or target_metric
        or parsed.get("mart_name")
        or parsed.get("validation_contract")
    )
    if not has_core_signal:
        raise ValueError("planner output missing core intent fields")

    return QuestionPlan(
        original_question=state["user_question"],
        route_kind=route_kind,
        question_type=question_type,
        task_type=task_type,
        requested_output=requested_output,
        target_metric=target_metric,
        dimensions=dimensions,
        filters=filters,
        time_condition=parsed.get("time_condition"),
        selected_join_tables=selected_tables,
        relevant_tables=relevant_tables,
        candidate_tables=candidate_tables,
        mart_name=parsed.get("mart_name"),
        grain=parsed.get("grain"),
        load_strategy=parsed.get("load_strategy"),
        ambiguity_note=parsed.get("ambiguity_note"),
        expected_result_shape=expected_result_shape,
        required_columns=required_columns,
        required_aggregations=required_aggregations,
        validation_contract=validation_contract,
        reasoning=str(parsed.get("reasoning") or ""),
    ).model_dump()


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

    return {"plan": normalized}

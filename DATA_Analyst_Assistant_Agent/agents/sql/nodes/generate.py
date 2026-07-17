"""generate_sql 노드: 질문 분석/마트 설계 기반 SQL 생성."""

from __future__ import annotations

import json
import logging
from typing import Any

from DATA_Analyst_Assistant_Agent.agents.sql import prompts
from DATA_Analyst_Assistant_Agent.agents.sql.planner_support import (
    empty_sql_draft,
    normalize_generated_sql,
    require_route_kind,
    retry_feedback_text,
    try_llm_json,
)
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState, SQLDraft
from DATA_Analyst_Assistant_Agent.agents.sql.generation_context import build_generation_context
from DATA_Analyst_Assistant_Agent.agents.sql.nodes.mart_design import (
    _mart_design_failure,
    validate_mart_design_state,
)


logger = logging.getLogger(__name__)


def _with_context_diagnostics(
    state: AgentState,
    update: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    history = [*(state.get("generation_context_diagnostics") or []), diagnostics]
    update["generation_context_diagnostics"] = history
    logger.info("SQL generation context diagnostics", extra={"diagnostics": diagnostics})
    return update


def _generation_failure(
    state: AgentState,
    *,
    reason_code: str,
    detail: str,
    retryable: bool,
) -> dict[str, Any]:
    suggested_action = "regenerate_sql" if retryable else "stop_and_surface_error"
    finding = {
        "category": "sql_generation_failed",
        "severity": "error",
        "retryable": retryable,
        "detail": detail,
        "message": detail,
        "source": "sql_generator",
        "code": reason_code,
        "suggested_action": suggested_action,
        "details": {"reason_code": reason_code},
    }
    retry_hint = {
        "retryable": retryable,
        "suggested_action": suggested_action,
        "reason_code": "sql_generation_failed",
        "details": {
            "generation_reason_code": reason_code,
            "messages": [detail],
        },
    }
    return {
        "sql_draft": empty_sql_draft(state, reasoning=detail),
        "generation_source": "failed",
        "generation_failure_reason": reason_code,
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


def generate_sql(state: AgentState):
    route_kind = require_route_kind(state["plan"])
    if route_kind == "comprehensive":
        try:
            validate_mart_design_state(state.get("mart_design") or {}, state.get("plan") or {})
        except Exception as exc:
            return _mart_design_failure(
                reason_code="stale_or_invalid_mart_design",
                detail=f"현재 계약을 만족하지 않는 mart 설계를 폐기하고 다시 설계해야 합니다: {exc}",
                retryable=True,
            )
    retry_hint = state.get("retry_hint") or {}
    if state.get("retry_count", 0) > 0 and retry_hint.get("reason_code") in {"missing_table", "missing_column"}:
        return _generation_failure(
            state,
            reason_code=str(retry_hint.get("reason_code") or "sql_generation_failed"),
            detail="스키마에 없는 테이블/컬럼 참조로 SQL 재생성이 무의미합니다.",
            retryable=False,
        )

    feedback = retry_feedback_text(state)
    try:
        context_result = build_generation_context(state, route_kind, feedback)
    except Exception as exc:
        return _generation_failure(
            state,
            reason_code="generation_context_invalid",
            detail=f"SQL 생성 컨텍스트를 구성하지 못했습니다: {type(exc).__name__}",
            retryable=True,
        )
    prompt = (
        prompts.generate_mart_prompt(context_result.context_json)
        if route_kind == "comprehensive"
        else prompts.generate_query_prompt(context_result.context_json)
    )
    diagnostics = dict(context_result.diagnostics)
    diagnostics["final_prompt_chars"] = len(prompt)
    response = try_llm_json(prompt)
    if not response:
        return _with_context_diagnostics(
            state,
            _generation_failure(
                state,
                reason_code="llm_empty_response",
                detail="LLM이 SQL 초안을 반환하지 않았습니다.",
                retryable=True,
            ),
            diagnostics,
        )

    cleaned = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return _with_context_diagnostics(
            state,
            _generation_failure(
                state,
                reason_code="llm_json_parse_failed",
                detail="LLM SQL 응답을 JSON으로 파싱하지 못했습니다.",
                retryable=True,
            ),
            diagnostics,
        )

    if not isinstance(parsed, dict):
        return _with_context_diagnostics(
            state,
            _generation_failure(
                state,
                reason_code="llm_json_not_object",
                detail="LLM SQL 응답이 JSON object가 아닙니다.",
                retryable=True,
            ),
            diagnostics,
        )

    try:
        normalized = normalize_generated_sql(
            parsed,
            route_kind,
            default_business_grain=(state.get("mart_design") or {}).get("grain") or state["plan"].get("grain"),
        )
    except ValueError as exc:
        return _with_context_diagnostics(
            state,
            _generation_failure(
                state,
                reason_code="sql_route_kind_mismatch",
                detail=str(exc),
                retryable=True,
            ),
            diagnostics,
        )

    return _with_context_diagnostics(
        state,
        {
            "sql_draft": SQLDraft(**normalized).model_dump(),
            "generation_source": "repair" if state.get("retry_count", 0) > 0 else "llm",
            "generation_failure_reason": "",
        },
        diagnostics,
    )

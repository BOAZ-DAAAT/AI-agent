"""generate_sql 노드: 질문 분석/마트 설계 기반 SQL 생성."""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import ValidationError

from DATA_Analyst_Assistant_Agent.agents.sql import prompts
from DATA_Analyst_Assistant_Agent.agents.sql.planner_support import (
    empty_sql_draft,
    normalize_generated_sql,
    require_route_kind,
    retry_feedback_text,
    invoke_llm_structured,
)
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState, SQLDraft
from DATA_Analyst_Assistant_Agent.agents.sql.generation_context import build_generation_context
from DATA_Analyst_Assistant_Agent.agents.sql.nodes.mart_design import (
    _mart_design_failure,
    validate_mart_design_state,
)


logger = logging.getLogger(__name__)


class _SimpleSQLDraft(SQLDraft):
    """simple route가 생성할 수 있는 SQLDraft 계약."""

    sql_type: Literal["select"] = "select"


class _ComprehensiveSQLDraft(SQLDraft):
    """comprehensive route가 생성할 수 있는 SQLDraft 계약."""

    sql_type: Literal["create_table_as"] = "create_table_as"


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
        prompts.generate_mart_prompt(context_result.context)
        if route_kind == "comprehensive"
        else prompts.generate_query_prompt(context_result.context)
    )
    diagnostics = dict(context_result.diagnostics)
    diagnostics["final_prompt_chars"] = len(prompt)
    response_model = _ComprehensiveSQLDraft if route_kind == "comprehensive" else _SimpleSQLDraft
    try:
        response = invoke_llm_structured(prompt, response_model)
    except ValidationError as exc:
        return _with_context_diagnostics(
            state,
            _generation_failure(
                state,
                reason_code="structured_output_contract_violation",
                detail=f"LLM SQL 응답이 structured output 계약을 만족하지 않습니다: {exc.errors()[0]['type']}",
                retryable=True,
            ),
            diagnostics,
        )
    except Exception as exc:
        return _with_context_diagnostics(
            state,
            _generation_failure(
                state,
                reason_code="structured_output_call_failed",
                detail=f"LLM structured output 호출에 실패했습니다: {type(exc).__name__}",
                retryable=True,
            ),
            diagnostics,
        )

    if response is None or response == "" or response == {}:
        return _with_context_diagnostics(
            state,
            _generation_failure(
                state,
                reason_code="structured_output_empty_response",
                detail="LLM이 SQL 초안을 반환하지 않았습니다.",
                retryable=True,
            ),
            diagnostics,
        )

    try:
        response_payload = response.model_dump() if isinstance(response, SQLDraft) else response
        parsed = response_model.model_validate(response_payload).model_dump()
    except ValidationError as exc:
        raw_sql_type = response.get("sql_type") if isinstance(response, dict) else None
        expected_sql_type = "create_table_as" if route_kind == "comprehensive" else "select"
        reason_code = (
            "sql_route_kind_mismatch"
            if raw_sql_type in {"select", "create_table_as"} and raw_sql_type != expected_sql_type
            else "structured_output_contract_violation"
        )
        detail = (
            f"{route_kind} route와 sql_type={raw_sql_type}이 일치하지 않습니다."
            if reason_code == "sql_route_kind_mismatch"
            else f"LLM SQL 응답이 structured output 계약을 만족하지 않습니다: {exc.errors()[0]['type']}"
        )
        return _with_context_diagnostics(
            state,
            _generation_failure(state, reason_code=reason_code, detail=detail, retryable=True),
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

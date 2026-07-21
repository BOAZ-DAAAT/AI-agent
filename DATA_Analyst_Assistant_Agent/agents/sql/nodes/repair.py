"""유효한 이전 SQLDraft를 최소 컨텍스트로 수정하는 repair_sql 노드."""

from __future__ import annotations

import json
from typing import Any

from DATA_Analyst_Assistant_Agent.agents.sql.planner_support import normalize_generated_sql, require_route_kind, try_llm_json
from DATA_Analyst_Assistant_Agent.agents.sql.prompts.repair import repair_sql_prompt
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState, SQLDraft


def _repair_failure(state: AgentState, reason_code: str, detail: str, *, retryable: bool = True) -> dict[str, Any]:
    finding = {
        "category": "sql_generation_failed",
        "severity": "error",
        "retryable": retryable,
        "detail": detail,
        "message": detail,
        "source": "sql_repair",
        "code": reason_code,
        "suggested_action": "repair_sql" if retryable else "stop_and_surface_error",
        "details": {"reason_code": reason_code, "generation_stage": "repair"},
    }
    retry_hint = {
        "retryable": retryable,
        "suggested_action": "repair_sql" if retryable else "stop_and_surface_error",
        "reason_code": "sql_generation_failed",
        "details": {
            "generation_stage": "repair",
            "generation_reason_code": reason_code,
            "messages": [detail],
            "original_retry_hint": state.get("retry_hint") or {},
        },
    }
    previous_findings = list(state.get("validation_findings") or [])
    findings = previous_findings + [finding]
    return {
        "sql_draft": {},
        "generation_source": "failed",
        "generation_failure_reason": reason_code,
        "validation": {
            "result": "invalid",
            "reason": detail,
            "feedback": detail,
            "findings": findings,
            "retry_hint": retry_hint,
        },
        "validation_findings": findings,
        "retry_hint": retry_hint,
        "feedback": detail,
        "error": detail,
    }


def repair_sql(state: AgentState) -> dict[str, Any]:
    previous_draft = state.get("previous_sql_draft") or state.get("sql_draft") or {}
    if not str(previous_draft.get("sql") or "").strip():
        return _repair_failure(
            state,
            "missing_previous_sql_draft",
            "repair할 유효한 이전 SQLDraft가 없습니다.",
            retryable=False,
        )

    response = try_llm_json(repair_sql_prompt(state, previous_draft))
    if not response or not response.strip():
        return _repair_failure(state, "llm_empty_response", "LLM이 SQL repair 초안을 반환하지 않았습니다.")
    cleaned = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return _repair_failure(state, "llm_json_parse_failed", "LLM SQL repair 응답을 JSON으로 파싱하지 못했습니다.")
    if not isinstance(parsed, dict):
        return _repair_failure(state, "llm_json_not_object", "LLM SQL repair 응답이 JSON object가 아닙니다.")

    route_kind = require_route_kind(state.get("plan") or {})
    try:
        normalized = normalize_generated_sql(
            parsed,
            route_kind,
            default_business_grain=(state.get("mart_design") or {}).get("grain") or (state.get("plan") or {}).get("grain"),
        )
        repaired_draft = SQLDraft(**normalized).model_dump()
        if not str(repaired_draft.get("sql") or "").strip().strip(";"):
            raise ValueError("repair SQL이 비어 있습니다")
    except Exception as exc:
        return _repair_failure(state, "repair_contract_invalid", f"SQL repair 응답이 SQLDraft 계약을 만족하지 않습니다: {exc}")

    return {
        "sql_draft": repaired_draft,
        "generation_source": "repair",
        "generation_failure_reason": "",
        "validation": {},
        "validation_findings": [],
        "retry_hint": {},
        "feedback": "",
        "error": "",
        "statement_results": [],
        "failed_sql_component": None,
        "failed_statement_index": None,
        "failed_statement_sql": "",
        "execution_error_info": {},
    }

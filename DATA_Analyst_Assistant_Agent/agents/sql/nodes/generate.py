"""generate_sql 노드: 질문 분석/마트 설계 기반 SQL 생성."""

from __future__ import annotations

import json

from DATA_Analyst_Assistant_Agent.agents.sql import prompts
from DATA_Analyst_Assistant_Agent.agents.sql.planner_support import (
    deterministic_sql_draft,
    normalize_generated_sql,
    retry_feedback_text,
    try_llm_json,
)
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState, SQLDraft


def generate_sql(state: AgentState):
    task_type = state["plan"].get("task_type", "query_answer")
    route_kind = state["plan"].get("route_kind") or ("comprehensive" if task_type == "data_mart_build" else "simple")
    fallback = deterministic_sql_draft(state)
    retry_hint = state.get("retry_hint") or {}
    # missing_table/missing_column: 스키마에 없는 대상 → LLM도 해결 불가 → 즉시 fallback
    # invalid_join_plan 등은 replan 경로 또는 강화된 feedback으로 LLM 재시도
    _HARD_FALLBACK_CODES = {"missing_table", "missing_column"}
    if state.get("retry_count", 0) > 0 and retry_hint.get("reason_code") in _HARD_FALLBACK_CODES:
        return {
            "sql_draft": fallback,
            "generation_source": "hard_fallback",
            "fallback_reason": retry_hint.get("reason_code", "hard_fallback"),
        }
    feedback = retry_feedback_text(state)
    prompt = prompts.generate_mart_prompt(state, feedback) if task_type == "data_mart_build" else prompts.generate_query_prompt(state, feedback)
    response = try_llm_json(prompt)
    generation_source = "llm"
    fallback_reason = ""
    if not response:
        parsed = fallback
        generation_source = "fallback"
        fallback_reason = "llm_empty_response"
    else:
        cleaned = response.strip().replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                parsed = fallback
                generation_source = "fallback"
                fallback_reason = "llm_json_not_object"
        except Exception:
            parsed = fallback
            generation_source = "fallback"
            fallback_reason = "llm_json_parse_failed"
    normalized = normalize_generated_sql(parsed, fallback, route_kind)
    return {
        "sql_draft": SQLDraft(**normalized).model_dump(),
        "generation_source": generation_source,
        "fallback_reason": fallback_reason,
    }

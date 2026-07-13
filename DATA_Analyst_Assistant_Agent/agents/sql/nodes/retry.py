"""increase_retry 노드: 재시도 카운트 증가 및 피드백 state 유지."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState


def increase_retry(state: AgentState):
    validation = state.get("validation") or {}
    retry_hint = state.get("retry_hint") or validation.get("retry_hint", {})
    return {
        "retry_count": state["retry_count"] + 1,
        # 직전 검증 피드백을 명시적으로 state에 유지.
        # retry_feedback_text()가 generate_sql에서 이 값을 prompt에 포함한다.
        "feedback": validation.get("feedback", ""),
        "validation": {},
        "validation_findings": [],
        # 다음 라우팅(route_after_retry)과 재시도 프롬프트 구성을 위해 reason_code를 유지한다.
        "retry_hint": retry_hint,
        "error": "",
        "failed_statement_index": None,
        "failed_statement_sql": "",
    }

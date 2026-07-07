"""increase_retry 노드: 재시도 카운트 증가 및 피드백 state 유지."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState


def increase_retry(state: AgentState):
    validation = state.get("validation") or {}
    return {
        "retry_count": state["retry_count"] + 1,
        # 직전 검증 피드백을 명시적으로 state에 유지.
        # retry_feedback_text()가 generate_sql에서 이 값을 prompt에 포함한다.
        "feedback": validation.get("feedback", ""),
    }

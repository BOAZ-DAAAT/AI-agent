"""increase_retry 노드: 재시도 카운트 증가."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState


def increase_retry(state: AgentState):
    return {"retry_count": state["retry_count"] + 1}

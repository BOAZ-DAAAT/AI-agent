"""relationship 노드 — 변수 간 관계 탐색."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import append_errors
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.react import RELATIONSHIP_TOOLS, run_mini_react_with_retry
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import relationship_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState


def relationship_node(state: EDAState) -> dict:
    prompt = relationship_prompt(state["user_question"], state.get("inspect_result", ""), state.get("analysis_plan", {}))
    result, err = run_mini_react_with_retry(RELATIONSHIP_TOOLS, prompt, "relationship")
    return {"relationship_result": result, "error_log": append_errors(state, err)}

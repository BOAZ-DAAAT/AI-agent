"""comparison 노드 — 그룹 간 비교 분석."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import append_errors
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.react import COMPARISON_TOOLS, run_mini_react_with_retry
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import comparison_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState


def comparison_node(state: EDAState) -> dict:
    prompt = comparison_prompt(state["user_question"], state.get("inspect_result", ""), state.get("analysis_plan", {}))
    result, err = run_mini_react_with_retry(COMPARISON_TOOLS, prompt, "comparison")
    return {"comparison_result": result, "error_log": append_errors(state, err)}

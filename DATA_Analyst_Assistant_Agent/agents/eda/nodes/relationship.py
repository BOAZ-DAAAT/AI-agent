"""relationship 노드 — 변수 간 관계 탐색."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import append_errors, get_context
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.tool_runner import run_tool_with_summary
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import relationship_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState
from DATA_Analyst_Assistant_Agent.agents.eda.tools import run_relationship


def relationship_node(state: EDAState) -> dict:
    ctx = get_context()
    if ctx.df is None:
        return {"relationship_result": "데이터가 로드되지 않았습니다.", "error_log": state.get("error_log", [])}

    plan = state.get("analysis_plan", {})
    result, err = run_tool_with_summary(
        run_relationship,
        lambda result_json: relationship_prompt(
            state["user_question"], state.get("inspect_result", ""), plan, result_json),
        "relationship",
    )
    return {"relationship_result": result, "error_log": append_errors(state, err)}

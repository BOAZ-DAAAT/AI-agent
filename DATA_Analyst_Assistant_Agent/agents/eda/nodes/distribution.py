"""distribution 노드 — 단변량 분포 분석."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import append_errors, get_context
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.tool_runner import run_tool_with_summary
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import distribution_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState
from DATA_Analyst_Assistant_Agent.agents.eda.tools import run_distribution


def distribution_node(state: EDAState) -> dict:
    ctx = get_context()
    if ctx.df is None:
        return {"distribution_result": "데이터가 로드되지 않았습니다.", "error_log": state.get("error_log", [])}

    plan = state.get("analysis_plan", {})
    result, err = run_tool_with_summary(
        run_distribution,
        lambda result_json: distribution_prompt(
            state["user_question"], state.get("inspect_result", ""), plan, result_json),
        "distribution",
    )
    return {"distribution_result": result, "error_log": append_errors(state, err)}

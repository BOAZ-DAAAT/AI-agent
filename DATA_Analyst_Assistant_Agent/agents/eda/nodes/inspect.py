"""inspect 노드 — 데이터 구조 파악."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import append_errors, get_context
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.tool_runner import run_tool_with_summary
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import inspect_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState
from DATA_Analyst_Assistant_Agent.agents.eda.tools import profile_data


def inspect_node(state: EDAState) -> dict:
    ctx = get_context()
    if ctx.df is None:
        return {"inspect_result": "데이터가 로드되지 않았습니다.", "error_log": state.get("error_log", [])}

    result, err = run_tool_with_summary(
        profile_data,
        lambda result_json: inspect_prompt(
            state["user_question"], state.get("mart_design", {}).get("grain", "미정"), result_json),
        "inspect",
    )
    return {"inspect_result": result, "error_log": append_errors(state, err)}

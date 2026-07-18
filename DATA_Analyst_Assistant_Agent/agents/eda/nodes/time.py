"""time 노드 — 시계열 추세/시즌성 분석."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import append_errors, get_context
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.tool_runner import run_tool_with_summary_and_facts
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import time_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState
from DATA_Analyst_Assistant_Agent.agents.eda.tools import run_time


def time_node(state: EDAState) -> dict:
    ctx = get_context()
    if ctx.df is None:
        return {"time_result": "데이터가 로드되지 않았습니다.", "time_facts": [],
                "error_log": state.get("error_log", [])}

    result, facts, err = run_tool_with_summary_and_facts(
        run_time,
        lambda result_json: time_prompt(state["user_question"], state.get("inspect_result", ""), result_json),
        "time",
    )
    return {"time_result": result, "time_facts": facts, "error_log": append_errors(state, err)}

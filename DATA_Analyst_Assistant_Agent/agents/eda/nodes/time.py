"""time 노드 — 시계열 추세/시즌성 분석."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import append_errors
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.react import TIME_TOOLS, run_mini_react_with_retry
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import time_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState


def time_node(state: EDAState) -> dict:
    prompt = time_prompt(state["user_question"], state.get("inspect_result", ""))
    result, err = run_mini_react_with_retry(TIME_TOOLS, prompt, "time")
    return {"time_result": result, "error_log": append_errors(state, err)}

"""distribution 노드 — 단변량 분포 분석."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import append_errors
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.react import DISTRIBUTION_TOOLS, run_mini_react_with_retry
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import distribution_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState


def distribution_node(state: EDAState) -> dict:
    prompt = distribution_prompt(state["user_question"], state.get("inspect_result", ""), state.get("analysis_plan", {}))
    result, err = run_mini_react_with_retry(DISTRIBUTION_TOOLS, prompt, "distribution")
    return {"distribution_result": result, "error_log": append_errors(state, err)}

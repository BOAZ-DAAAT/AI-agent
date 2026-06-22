"""quality 노드 — 데이터 품질 점검."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import append_errors
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.react import QUALITY_TOOLS, run_mini_react_with_retry
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import quality_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState


def quality_node(state: EDAState) -> dict:
    prompt = quality_prompt(state["user_question"], state.get("inspect_result", ""))
    result, err = run_mini_react_with_retry(QUALITY_TOOLS, prompt, "quality")
    return {"quality_result": result, "error_log": append_errors(state, err)}

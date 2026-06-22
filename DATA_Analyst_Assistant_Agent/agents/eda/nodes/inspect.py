"""inspect 노드 — 데이터 구조 파악."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import append_errors
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.react import INSPECT_TOOLS, run_mini_react_with_retry
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import inspect_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState


def inspect_node(state: EDAState) -> dict:
    prompt = inspect_prompt(state["user_question"], state.get("mart_design", {}).get("grain", "미정"))
    result, err = run_mini_react_with_retry(INSPECT_TOOLS, prompt, "inspect")
    return {"inspect_result": result, "error_log": append_errors(state, err)}

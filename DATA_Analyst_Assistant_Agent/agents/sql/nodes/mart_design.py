"""design_mart 노드: 데이터마트 설계(task_type=data_mart_build 일 때만)."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql import prompts
from DATA_Analyst_Assistant_Agent.agents.sql.planner_support import default_mart_design, try_llm_json
from DATA_Analyst_Assistant_Agent.agents.sql._runtime import safe_json_parse
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState, MartDesign


def design_mart(state: AgentState):
    if state["plan"].get("task_type") != "data_mart_build":
        return {"mart_design": {}}

    fallback = default_mart_design(state)
    response = try_llm_json(prompts.mart_design_prompt(state))
    parsed = safe_json_parse(response, fallback) if response else fallback
    merged = dict(fallback)
    merged.update({k: v for k, v in parsed.items() if v not in (None, "", [], {})})
    return {"mart_design": MartDesign(**merged).model_dump()}

"""plan_question 노드: 사용자 질문 분석."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql import prompts
from DATA_Analyst_Assistant_Agent.agents.sql.planner_support import default_plan_from_state, try_llm_json
from DATA_Analyst_Assistant_Agent.agents.sql._runtime import safe_json_parse
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState, QuestionPlan


def plan_question(state: AgentState):
    fallback = default_plan_from_state(state)
    response = try_llm_json(prompts.plan_prompt(state))
    parsed = safe_json_parse(response, fallback) if response else fallback
    merged = dict(fallback)
    merged.update({k: v for k, v in parsed.items() if v not in (None, "", [], {})})
    parsed = QuestionPlan(**merged).model_dump()
    parsed["original_question"] = state["user_question"]
    return {"plan": parsed}

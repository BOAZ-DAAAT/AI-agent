"""generate_sql 노드: 질문 분석/마트 설계 기반 SQL 생성."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql import prompts
from DATA_Analyst_Assistant_Agent.agents.sql._runtime import safe_json_parse
from DATA_Analyst_Assistant_Agent.agents.sql.planner_support import (
    deterministic_sql_draft,
    normalize_generated_sql,
    retry_feedback_text,
    try_llm_json,
)
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState, SQLDraft


def generate_sql(state: AgentState):
    task_type = state["plan"].get("task_type", "query_answer")
    route_kind = state["plan"].get("route_kind") or ("comprehensive" if task_type == "data_mart_build" else "simple")
    fallback = deterministic_sql_draft(state)
    feedback = retry_feedback_text(state)
    prompt = prompts.generate_mart_prompt(state, feedback) if task_type == "data_mart_build" else prompts.generate_query_prompt(state, feedback)
    response = try_llm_json(prompt)
    parsed = safe_json_parse(response, fallback) if response else fallback
    normalized = normalize_generated_sql(parsed, fallback, route_kind)
    return {"sql_draft": SQLDraft(**normalized).model_dump()}

"""plan_question 노드: 사용자 질문 분석."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql import prompts
from DATA_Analyst_Assistant_Agent.agents.sql._runtime import get_llm, safe_json_parse
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState, QuestionPlan


def plan_question(state: AgentState):
    response = get_llm().invoke(prompts.plan_prompt(state)).content

    fallback = QuestionPlan(
        original_question=state["user_question"],
        question_type="unknown",
        task_type="query_answer",
        requested_output="execute_and_answer",
        target_metric="unknown",
        dimensions=[],
        filters=[],
        time_condition=None,
        relevant_tables=[],
        mart_name=None,
        grain=None,
        load_strategy=None,
        ambiguity_note="질문 분석 파싱 실패"
    ).model_dump()

    parsed = safe_json_parse(response, fallback)
    parsed["original_question"] = state["user_question"]

    return {"plan": parsed}

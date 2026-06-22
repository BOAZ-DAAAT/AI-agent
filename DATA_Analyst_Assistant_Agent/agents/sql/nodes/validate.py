"""validate_sql_and_result 노드: 실행 결과 검증."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql import prompts
from DATA_Analyst_Assistant_Agent.agents.sql._runtime import get_llm, safe_json_parse
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState, ValidationResult


def validate_sql_and_result(state: AgentState):
    if state.get("error"):
        parsed = ValidationResult(
            result="invalid",
            reason=f"SQL 실행 오류: {state['error']}",
            feedback="실행 오류를 해결하고 task_type에 맞는 MySQL SQL로 다시 생성하라."
        ).model_dump()
        return {"validation": parsed, "feedback": parsed["feedback"]}

    response = get_llm().invoke(prompts.validate_prompt(state)).content

    fallback = ValidationResult(
        result="invalid",
        reason="검증 결과 파싱 실패",
        feedback="질문 조건, grain, 정합성 점검 내용을 반영해 다시 SQL을 생성하라."
    ).model_dump()

    parsed = safe_json_parse(response, fallback)
    return {"validation": parsed, "feedback": parsed.get("feedback", "")}

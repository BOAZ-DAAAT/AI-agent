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

    mart_signal = _validate_mart_reusability(state)

    prompt_state = dict(state)
    if mart_signal is not None:
        prompt_state["mart_validation_signal"] = mart_signal

    response = get_llm().invoke(prompts.validate_prompt(prompt_state)).content

    fallback = ValidationResult(
        result="invalid",
        reason="검증 결과 파싱 실패",
        feedback="질문 조건, grain, 정합성 점검 내용을 반영해 다시 SQL을 생성하라."
    ).model_dump()

    parsed = safe_json_parse(response, fallback)
    if mart_signal is not None:
        signal_reason = mart_signal.get("reason", "")
        signal_feedback = mart_signal.get("feedback", "")
        if parsed.get("result") == "valid":
            parsed["feedback"] = parsed.get("feedback") or signal_feedback
        else:
            if signal_reason and signal_reason not in (parsed.get("reason") or ""):
                parsed["reason"] = f"{parsed.get('reason', '')} / 참고 신호: {signal_reason}".strip(" /")
            if signal_feedback and signal_feedback not in (parsed.get("feedback") or ""):
                parsed["feedback"] = f"{parsed.get('feedback', '')} {signal_feedback}".strip()
    return {"validation": parsed, "feedback": parsed.get("feedback", "")}


def _validate_mart_reusability(state: AgentState) -> dict | None:
    if state["plan"].get("task_type") != "data_mart_build":
        return None

    sql = (state.get("sql_draft") or {}).get("sql", "")
    sql_upper = sql.upper()
    has_aggregate_summary = any(token in sql_upper for token in ("GROUP BY", "HAVING", "COUNT(", "SUM(", "AVG(", "MIN(", "MAX("))
    if not has_aggregate_summary:
        return None

    mart_design = state.get("mart_design") or {}
    rationale = " ".join(
        str(value or "")
        for value in (
            mart_design.get("aggregation_rationale"),
            mart_design.get("design_reasoning"),
            (state.get("sql_draft") or {}).get("reasoning"),
        )
    )
    if any(token in rationale for token in ("row-level", "row level", "원본 행", "행 수준", "불가피", "정당화", "예외")):
        return None

    return {
        "severity": "warning",
        "reason": "데이터마트 SQL이 최종 요약 결과 테이블처럼 과하게 집계된 신호가 있습니다.",
        "feedback": (
            "가능하면 최종 요약 결과 대신 재사용 가능한 상세 기반 테이블로 다시 작성하세요. "
            "원본 entity/event 행 수준 grain 유지, 조인/정제/표준화/필수 파생 컬럼 추가를 우선 검토하세요. "
            "집계가 정말 필요하다면 왜 row-level mart가 부적절한지 reasoning에 명시하세요."
        ),
    }

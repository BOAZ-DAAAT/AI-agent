"""increase_retry 노드: 재시도 카운트 증가 및 피드백 state 유지."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState


REPAIR_REASON_CODES = {
    "mysql_dialect_error",
    "route_kind_mismatch",
    "intent_mismatch",
    "result_shape_mismatch",
    "mart_policy_mismatch",
    "mart_summary_bias",
    "execution_error",
}


def _is_repair_retry(retry_hint: dict) -> bool:
    reason_code = retry_hint.get("reason_code")
    if reason_code in REPAIR_REASON_CODES:
        return True
    return reason_code == "sql_generation_failed" and (retry_hint.get("details") or {}).get("generation_stage") == "repair"


def increase_retry(state: AgentState):
    validation = state.get("validation") or {}
    retry_hint = state.get("retry_hint") or validation.get("retry_hint", {})
    repair_retry = _is_repair_retry(retry_hint)
    update = {
        "retry_count": state["retry_count"] + 1,
        # 직전 검증 피드백을 명시적으로 state에 유지.
        # retry_feedback_text()가 generate_sql에서 이 값을 prompt에 포함한다.
        "feedback": validation.get("feedback", ""),
        # 다음 라우팅(route_after_retry)과 재시도 프롬프트 구성을 위해 reason_code를 유지한다.
        "retry_hint": retry_hint,
    }
    if repair_retry:
        current_draft = state.get("sql_draft") or {}
        previous_draft = state.get("previous_sql_draft") or {}
        if str(current_draft.get("sql") or "").strip():
            previous_draft = current_draft
        update.update({
            "previous_sql_draft": previous_draft,
            "sql_draft": {},
        })
        return update

    update.update({
        "validation": {},
        "validation_findings": [],
        "error": "",
        "failed_sql_component": None,
        "failed_statement_index": None,
        "failed_statement_sql": "",
        "execution_error_info": {},
    })
    reason_code = retry_hint.get("reason_code")
    if reason_code == "sql_plan_failed":
        update.update({
            "question_plan": {},
            "final_table_plan": {},
            "planning_stages": {},
            "plan": {},
            "mart_design": {},
            "sql_draft": {},
        })
    elif reason_code in {"sql_mart_design_failed", "mart_grain_missing", "mart_column_contract_invalid"}:
        update.update({
            "mart_design": {},
            "sql_draft": {},
        })
    else:
        update["sql_draft"] = {}
    return update

"""validate_sql_and_result 노드: 구조화된 실행 결과 검증."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql._runtime import format_result_rows
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState
from DATA_Analyst_Assistant_Agent.agents.sql.validation_contract import summarize_validation, validate_result_shape


def validate_sql_and_result(state: AgentState):
    existing_findings = list(state.get("validation_findings") or [])
    existing_validation = state.get("validation") or {}
    if existing_validation.get("result") == "invalid" and existing_findings:
        summary = summarize_validation(existing_findings)
        return {"validation": summary, "validation_findings": existing_findings, "retry_hint": summary.get("retry_hint", {}), "feedback": summary.get("feedback", "")}
    if state.get("error"):
        error_info = dict(state.get("execution_error_info") or {})
        statement_index = state.get("failed_statement_index")
        statement_sql = (state.get("failed_statement_sql") or "").strip()
        component = state.get("failed_sql_component") or error_info.get("component") or "main"
        prefix = f"{statement_index + 1}번 statement 실행 오류" if statement_index is not None else f"{component} SQL 실행 오류"
        detail = f"{prefix}: {state['error']}"
        if statement_sql:
            detail += f" | 실패 SQL: {statement_sql}"
        classification = str(error_info.get("classification") or "unknown")
        category = "missing_table" if classification == "missing_table" else "execution_error"
        retryable = classification == "repairable_sql" and bool(error_info.get("retryable"))
        finding = {
            "category": category,
            "severity": "error",
            "retryable": retryable,
            "detail": detail,
            "code": error_info.get("error_code") or classification,
            "suggested_action": "repair_sql" if retryable else "stop_and_surface_error",
            "details": error_info,
        }
        findings = existing_findings + [finding]
        summary = summarize_validation(findings)
        if not summary.get("feedback"):
            summary["feedback"] = f"실행 오류를 해결하도록 SQL을 다시 작성하세요. 실패 원인: {state['error']}."
        return {"validation": summary, "validation_findings": findings, "retry_hint": summary.get("retry_hint", {}), "feedback": summary.get("feedback", "")}
    findings = list(existing_findings)
    findings.extend(
        validate_result_shape(
            state.get("plan", {}),
            state.get("sql_draft", {}),
            state.get("sql_result"),
            int(state.get("row_count", 0) or 0),
            state.get("postcheck_result"),
        )
    )
    summary = summarize_validation(findings)
    if summary["result"] == "valid":
        summary["reason"] = f"구조화 검증 통과. 결과 미리보기: {format_result_rows(state.get('sql_result'), max_rows=3)}"
    return {"validation": summary, "validation_findings": findings, "retry_hint": summary.get("retry_hint", {}), "feedback": summary.get("feedback", "")}

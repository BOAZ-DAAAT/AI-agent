from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState
from DATA_Analyst_Assistant_Agent.agents.sql.validation_contract import (
    summarize_validation,
    validate_datamart_reusability,
    validate_sql_dialect_and_route,
    validate_sql_identifiers,
    validate_sql_intent,
)


def prevalidate_sql(state: AgentState):
    existing_findings = list(state.get("validation_findings") or [])
    existing_validation = state.get("validation") or {}
    if existing_validation.get("result") == "invalid" and existing_findings:
        return {
            "validation": existing_validation,
            "validation_findings": existing_findings,
            "retry_hint": (state.get("retry_hint") or existing_validation.get("retry_hint", {})),
            "feedback": existing_validation.get("feedback", "") or state.get("feedback", ""),
            "error": existing_validation.get("reason", "") or state.get("error", ""),
        }
    plan = state.get("plan", {})
    sql_draft = state.get("sql_draft", {})
    findings = []
    findings.extend(validate_sql_dialect_and_route(plan, sql_draft))
    findings.extend(validate_sql_intent(plan, sql_draft))
    findings.extend(validate_sql_identifiers(plan, sql_draft, state.get("schema_text", "")))
    findings.extend(validate_datamart_reusability(plan, state.get("mart_design", {}), sql_draft))
    summary = summarize_validation(findings)
    return {
        "validation": summary,
        "validation_findings": findings,
        "retry_hint": summary.get("retry_hint", {}),
        "feedback": summary.get("feedback", ""),
        "error": summary.get("reason", "") if summary.get("result") == "invalid" else "",
    }


def route_after_prevalidation(state: AgentState):
    if (state.get("validation") or {}).get("result") == "invalid":
        return "validate"
    return "execute"

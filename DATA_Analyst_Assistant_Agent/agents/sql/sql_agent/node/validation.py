from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql.sql_agent.tool.runtime import AgentState
from DATA_Analyst_Assistant_Agent.agents.sql.sql_agent.tool.validation_contract import (
    summarize_validation,
    validate_sql_dialect_and_route,
    validate_sql_identifiers,
    validate_sql_intent,
)


def prevalidate_sql(state: AgentState):
    plan = state.get("plan", {})
    sql_draft = state.get("sql_draft", {})
    schema_text = state.get("schema_text", "")
    findings = []
    findings.extend(validate_sql_dialect_and_route(plan, sql_draft))
    findings.extend(validate_sql_intent(plan, sql_draft))
    findings.extend(validate_sql_identifiers(plan, sql_draft, schema_text))
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

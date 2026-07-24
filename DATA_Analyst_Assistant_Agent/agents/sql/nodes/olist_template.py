"""Olist 결정론적 SQL 초안 생성 노드."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql.olist_templates import (
    build_olist_mart_design,
    build_olist_sql_draft,
    build_olist_validation_plan,
)
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState
from DATA_Analyst_Assistant_Agent.shared.contracts import OlistTemplateId


def build_olist_template_sql(state: AgentState) -> dict:
    template_value = state.get("sql_template_id")
    try:
        template_id = OlistTemplateId(str(template_value or ""))
        draft = build_olist_sql_draft(
            template_id,
            state.get("required_db_schema") or None,
            parameters=state.get("sql_template_parameters") or {},
        )
        plan = build_olist_validation_plan(template_id)
        mart_design = build_olist_mart_design(template_id)
    except Exception as exc:
        detail = f"Olist SQL 템플릿 구성에 실패했습니다: {exc}"
        finding = {
            "category": "olist_template_failure",
            "code": "olist_template_build_failed",
            "source": "olist_template",
            "severity": "error",
            "disposition": "error",
            "retryable": False,
            "suggested_action": "stop_and_surface_error",
            "detail": detail,
            "message": detail,
        }
        retry_hint = {
            "retryable": False,
            "suggested_action": "stop_and_surface_error",
            "reason_code": "olist_template_failure",
            "details": {"failure_reason": detail},
        }
        return {
            "generation_source": "failed",
            "sql_generation_source": "failed",
            "generation_failure_reason": "olist_template_build_failed",
            "validation": {
                "result": "invalid",
                "reason": detail,
                "feedback": detail,
                "findings": [finding],
                "retry_hint": retry_hint,
            },
            "validation_findings": [finding],
            "retry_hint": retry_hint,
            "error": detail,
        }

    return {
        "schema_text": state.get("required_db_schema") or state.get("schema_text", ""),
        "plan": plan,
        "question_plan": plan,
        "final_table_plan": {
            "selected_join_tables": plan["selected_join_tables"],
            "required_columns": plan["required_columns"],
        },
        "planning_stages": {"olist_template": template_id.value},
        "mart_design": mart_design,
        "sql_draft": draft.model_dump(),
        "generation_source": "olist_template",
        "sql_generation_source": "olist_template",
        "generation_failure_reason": "",
        "validation": {},
        "validation_findings": [],
        "retry_hint": {},
        "error": "",
    }


__all__ = ["build_olist_template_sql"]

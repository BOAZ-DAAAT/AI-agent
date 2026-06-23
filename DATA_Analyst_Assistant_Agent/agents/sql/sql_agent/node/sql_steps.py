from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql.sql_agent.prompt.templates import sql_generation_prompt
from DATA_Analyst_Assistant_Agent.agents.sql.sql_agent.tool.planner_support import (
    deterministic_sql_draft,
    normalize_generated_sql,
    retry_feedback_text,
    safe_json_parse,
    try_llm_json,
)
from DATA_Analyst_Assistant_Agent.agents.sql.sql_agent.tool.runtime import AgentState
from DATA_Analyst_Assistant_Agent.agents.sql.sql_agent.tool.sql_utils import (
    can_use_live_db,
    format_result_rows,
    is_safe_mart_sql,
    is_safe_query_sql,
    offline_mart_rows,
    offline_select_rows,
    run_sql_commit,
    run_sql_fetchall,
    validate_mysql_sql,
)
from DATA_Analyst_Assistant_Agent.agents.sql.sql_agent.tool.validation_contract import (
    summarize_validation,
    validate_result_shape,
)


def generate_sql(state: AgentState):
    feedback = retry_feedback_text(state)
    plan = state["plan"]
    fallback = deterministic_sql_draft(state)
    retry_hint = state.get("retry_hint") or {}
    if state.get("retry_count", 0) > 0 and retry_hint.get("reason_code") in {
        "missing_table",
        "missing_column",
        "invalid_join_plan",
    }:
        return {"sql_draft": fallback}
    response = try_llm_json(sql_generation_prompt(state, feedback))
    parsed = safe_json_parse(response, fallback) if response else fallback
    parsed = normalize_generated_sql(parsed, fallback, plan.get("route_kind", "simple"))
    return {"sql_draft": parsed}


def execute_sql(state: AgentState):
    sql = state["sql_draft"]["sql"].strip()
    sql_type = state["sql_draft"].get("sql_type", "select")
    target_table = state["sql_draft"].get("target_table")

    try:
        dialect_issue = validate_mysql_sql(sql)
        if dialect_issue:
            return {
                "sql_result": None,
                "row_count": 0,
                "precheck_result": None,
                "postcheck_result": None,
                "error": f"MySQL 문법 호환성 검사 실패: {dialect_issue}",
            }

        pre_rows = None
        if state["sql_draft"].get("precheck_sql") and can_use_live_db():
            pre_rows = run_sql_fetchall(state["sql_draft"]["precheck_sql"])

        if sql_type == "select":
            if not is_safe_query_sql(sql):
                return {
                    "sql_result": None,
                    "row_count": 0,
                    "precheck_result": pre_rows,
                    "postcheck_result": None,
                    "error": "조회 SQL 안전성 검사 실패",
                }

            rows = run_sql_fetchall(sql) if can_use_live_db() else offline_select_rows(sql)
            return {
                "sql_result": rows,
                "row_count": len(rows),
                "precheck_result": pre_rows,
                "postcheck_result": None,
                "error": "",
            }

        ok, reason = is_safe_mart_sql(sql, target_table)
        if not ok:
            return {
                "sql_result": None,
                "row_count": 0,
                "precheck_result": pre_rows,
                "postcheck_result": None,
                "error": reason,
            }

        if can_use_live_db():
            run_sql_commit(sql)
        post_rows = None
        if state["sql_draft"].get("postcheck_sql") and can_use_live_db():
            post_rows = run_sql_fetchall(state["sql_draft"]["postcheck_sql"])

        return {
            "sql_result": [("datamart 생성 완료", target_table)] if can_use_live_db() else offline_mart_rows(target_table),
            "row_count": 1,
            "precheck_result": pre_rows,
            "postcheck_result": post_rows,
            "error": "",
        }
    except Exception as e:
        return {
            "sql_result": None,
            "row_count": 0,
            "precheck_result": None,
            "postcheck_result": None,
            "error": str(e),
        }


def validate_sql_and_result(state: AgentState):
    existing_findings = list(state.get("validation_findings") or [])
    existing_validation = state.get("validation") or {}
    if existing_validation.get("result") == "invalid" and existing_findings:
        summary = summarize_validation(existing_findings)
        return {
            "validation": summary,
            "validation_findings": existing_findings,
            "retry_hint": summary.get("retry_hint", {}),
            "feedback": summary.get("feedback", ""),
        }

    if state.get("error"):
        findings = existing_findings + [
            {
                "category": "execution_error",
                "severity": "error",
                "retryable": True,
                "detail": f"SQL 실행 오류: {state['error']}",
            }
        ]
        summary = summarize_validation(findings)
        if not summary.get("feedback"):
            summary["feedback"] = (
                "실행 오류를 해결하도록 SQL을 다시 작성하세요. "
                f"실패 원인: {state['error']}. "
                "반드시 MySQL 8 문법만 사용하고, 조인 대상 테이블과 route_kind가 질문에 맞는지 다시 확인하세요."
            )
        return {
            "validation": summary,
            "validation_findings": findings,
            "retry_hint": summary.get("retry_hint", {}),
            "feedback": summary.get("feedback", ""),
        }

    plan = state.get("plan", {})
    sql_draft = state.get("sql_draft", {})
    route_kind = plan.get("route_kind", "simple")
    sql_type = sql_draft.get("sql_type", "select")

    findings = list(existing_findings)
    if route_kind == "simple" and sql_type != "select":
        findings.append(
            {
                "category": "route_kind_mismatch",
                "severity": "error",
                "retryable": True,
                "detail": "simple 경로인데 조회 SQL이 아닙니다.",
            }
        )
    if route_kind == "comprehensive" and sql_type == "select":
        findings.append(
            {
                "category": "route_kind_mismatch",
                "severity": "error",
                "retryable": True,
                "detail": "comprehensive 경로인데 datamart 생성 SQL이 아닙니다.",
            }
        )

    findings.extend(
        validate_result_shape(
            plan,
            sql_draft,
            state.get("sql_result"),
            int(state.get("row_count", 0) or 0),
            state.get("postcheck_result"),
        )
    )
    summary = summarize_validation(findings)
    if summary["result"] == "valid":
        summary["reason"] = "route_kind, SQL 유형, 결과 형태 검증을 모두 통과했습니다."
    return {
        "validation": summary,
        "validation_findings": findings,
        "retry_hint": summary.get("retry_hint", {}),
        "feedback": summary.get("feedback", ""),
    }


def finalize_answer(state: AgentState):
    sql_text = state.get("sql_draft", {}).get("sql", "")
    route_kind = state.get("plan", {}).get("route_kind", "simple")
    if state["validation"].get("result") != "valid":
        return {
            "final_answer": (
                "SQL 생성 또는 실행 검증에 실패했습니다.\n"
                f"사유: {state['validation'].get('reason')}\n"
                f"마지막 SQL: {sql_text}"
            )
        }

    if route_kind == "comprehensive":
        return {
            "final_answer": (
                "comprehensive 경로로 datamart 생성 SQL을 작성하고 직접 MySQL에 실행했습니다.\n"
                f"대상 테이블: {state.get('sql_draft', {}).get('target_table') or '미지정'}\n"
                f"실행 결과 미리보기: {format_result_rows(state.get('sql_result'), max_rows=5)}\n"
                f"최종 SQL:\n{sql_text}"
            )
        }

    return {
        "final_answer": (
            "simple 경로로 조회 SQL을 작성하고 직접 MySQL에 실행했습니다.\n"
            f"행 수: {state.get('row_count', 0)}\n"
            f"실행 결과 미리보기: {format_result_rows(state.get('sql_result'), max_rows=5)}\n"
            f"최종 SQL:\n{sql_text}"
        )
    }


def increase_retry(state: AgentState):
    return {"retry_count": state["retry_count"] + 1}


def route_after_validation(state: AgentState):
    if state["validation"].get("result") == "valid":
        return "finalize"
    if state["retry_count"] >= state["max_retries"]:
        return "finalize"
    return "retry"

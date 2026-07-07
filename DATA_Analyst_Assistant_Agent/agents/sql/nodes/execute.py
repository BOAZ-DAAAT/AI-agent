"""execute_sql 노드: 안전성 검사 후 SQL 실행(조회/마트)."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql._runtime import (
    can_use_live_db,
    drop_table_if_exists,
    ensure_target_schema_exists,
    offline_mart_rows,
    offline_select_rows,
    is_safe_mart_sql,
    is_safe_query_sql,
    run_sql_commit,
    run_sql_fetchall,
    validate_mysql_sql,
)
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState


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
                "error": f"MySQL 문법 호환성 검사 실패: {dialect_issue}"
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
                    "error": "조회 SQL 안전성 검사 실패"
                }

            rows = run_sql_fetchall(sql) if can_use_live_db() else offline_select_rows(sql)

            return {
                "sql_result": rows,
                "row_count": len(rows),
                "precheck_result": pre_rows,
                "postcheck_result": None,
                "error": ""
            }

        ok, reason = is_safe_mart_sql(sql, target_table)
        if not ok:
            return {
                "sql_result": None,
                "row_count": 0,
                "precheck_result": pre_rows,
                "postcheck_result": None,
                "error": reason
            }

        if can_use_live_db():
            try:
                ensure_target_schema_exists(target_table)
                run_sql_commit(sql)
            except Exception as e:
                error_text = str(e)
                if sql_type in {"create_table_as", "insert_select"} and target_table and "already exists" in error_text.lower():
                    drop_table_if_exists(target_table)
                    run_sql_commit(sql)
                else:
                    raise

        post_rows = None
        if state["sql_draft"].get("postcheck_sql") and can_use_live_db():
            post_rows = run_sql_fetchall(state["sql_draft"]["postcheck_sql"])

        return {
            "sql_result": [("마트 생성 완료", target_table)] if can_use_live_db() else offline_mart_rows(target_table),
            "row_count": 1,
            "precheck_result": pre_rows,
            "postcheck_result": post_rows,
            "error": ""
        }

    except Exception as e:
        return {
            "sql_result": None,
            "row_count": 0,
            "precheck_result": None,
            "postcheck_result": None,
            "error": str(e)
        }

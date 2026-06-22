"""execute_sql 노드: 안전성 검사 후 SQL 실행(조회/마트)."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql._runtime import (
    is_safe_mart_sql,
    is_safe_query_sql,
    run_sql_commit,
    run_sql_fetchall,
)
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState


def execute_sql(state: AgentState):
    sql = state["sql_draft"]["sql"].strip()
    sql_type = state["sql_draft"].get("sql_type", "select")
    target_table = state["sql_draft"].get("target_table")

    try:
        pre_rows = None
        if state["sql_draft"].get("precheck_sql"):
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

            rows = run_sql_fetchall(sql)

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

        run_sql_commit(sql)

        post_rows = None
        if state["sql_draft"].get("postcheck_sql"):
            post_rows = run_sql_fetchall(state["sql_draft"]["postcheck_sql"])

        return {
            "sql_result": [("마트 생성 완료", target_table)],
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

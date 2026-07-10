"""execute_sql 노드: 안전성 검사 후 SQL 실행(조회/마트)."""

from __future__ import annotations

from typing import Any

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
    split_sql_statements,
    validate_mysql_sql,
)
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState


def _statement_result(index: int, statement_sql: str, rows: list[Any]) -> dict[str, Any]:
    columns: list[str] = []
    if rows:
        first = rows[0]
        if hasattr(first, "_mapping"):
            columns = list(first._mapping.keys())
        elif isinstance(first, dict):
            columns = list(first.keys())
        elif isinstance(first, (list, tuple)):
            columns = [f"col_{idx + 1}" for idx, _ in enumerate(first)]
    return {
        "index": index,
        "sql": statement_sql,
        "row_count": len(rows),
        "columns": columns,
        "rows": rows,
    }


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
                    "statement_results": [],
                    "row_count": 0,
                    "precheck_result": pre_rows,
                    "postcheck_result": None,
                    "error": "조회 SQL 안전성 검사 실패"
                }

            statements = split_sql_statements(sql)
            statement_results: list[dict[str, Any]] = []
            for index, statement_sql in enumerate(statements):
                try:
                    rows = run_sql_fetchall(statement_sql) if can_use_live_db() else offline_select_rows(statement_sql)
                except Exception as e:
                    return {
                        "sql_result": statement_results[-1]["rows"] if statement_results else None,
                        "statement_results": statement_results,
                        "row_count": statement_results[-1]["row_count"] if statement_results else 0,
                        "precheck_result": pre_rows,
                        "postcheck_result": None,
                        "error": str(e),
                        "failed_statement_index": index,
                        "failed_statement_sql": statement_sql,
                    }
                statement_results.append(_statement_result(index, statement_sql, list(rows)))

            return {
                "sql_result": statement_results[-1]["rows"] if statement_results else None,
                "statement_results": statement_results,
                "row_count": statement_results[-1]["row_count"] if statement_results else 0,
                "precheck_result": pre_rows,
                "postcheck_result": None,
                "failed_statement_index": None,
                "failed_statement_sql": "",
                "error": ""
            }

        ok, reason = is_safe_mart_sql(sql, target_table)
        if not ok:
            return {
                "sql_result": None,
                "statement_results": [],
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
            "statement_results": [],
            "row_count": 1,
            "precheck_result": pre_rows,
            "postcheck_result": post_rows,
            "failed_statement_index": None,
            "failed_statement_sql": "",
            "error": ""
        }

    except Exception as e:
        return {
            "sql_result": None,
            "statement_results": [],
            "row_count": 0,
            "precheck_result": None,
            "postcheck_result": None,
            "failed_statement_index": state.get("failed_statement_index"),
            "failed_statement_sql": state.get("failed_statement_sql", ""),
            "error": str(e)
        }

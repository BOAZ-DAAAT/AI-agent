"""execute_sql 노드: 안전성 검사 후 SQL 실행(조회/마트)."""

from __future__ import annotations

import os
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
from DATA_Analyst_Assistant_Agent.agents.sql.execution_errors import classify_execution_error
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState

# 마트 생성 후 sql_result 에 담을 미리보기 행 수. 하류(EDA/분석)는 마트를 DB 로 직접
# 조회(전체)하므로 이 미리보기는 검증/최종답변 표시·CSV 폴백용 소량 샘플이다.
MART_PREVIEW_ROWS = int(os.getenv("MART_PREVIEW_ROWS", "100"))


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


def _execution_failure(
    error: BaseException | str,
    *,
    component: str,
    failed_sql: str,
    statement_index: int | None = None,
    statement_results: list[dict[str, Any]] | None = None,
    precheck_result: Any = None,
) -> dict[str, Any]:
    """실패 위치와 드라이버 오류 정보를 잃지 않는 공통 실행 실패 응답이다."""
    info = classify_execution_error(error)
    info.update({
        "component": component,
        "statement_index": statement_index,
        "statement_number": statement_index + 1 if statement_index is not None else None,
        "sql": failed_sql,
    })
    completed = list(statement_results or [])
    return {
        "sql_result": completed[-1]["rows"] if completed else None,
        "statement_results": completed,
        "row_count": completed[-1]["row_count"] if completed else 0,
        "precheck_result": precheck_result,
        "postcheck_result": None,
        "error": str(error),
        "failed_sql_component": component,
        "failed_statement_index": statement_index,
        "failed_statement_sql": failed_sql,
        "execution_error_info": info,
        "classification": info["classification"],
        "repair_strategy": info["repair_strategy"],
    }


def execute_sql(state: AgentState):
    sql = state["sql_draft"]["sql"].strip()
    sql_type = state["sql_draft"].get("sql_type", "select")
    target_table = state["sql_draft"].get("target_table")

    dialect_issue = validate_mysql_sql(sql)
    if dialect_issue:
        return _execution_failure(
            f"MySQL 문법 호환성 검사 실패: {dialect_issue}",
            component="main",
            failed_sql=sql,
        )

    pre_rows = None
    precheck_sql = str(state["sql_draft"].get("precheck_sql") or "").strip()
    if precheck_sql and can_use_live_db():
        try:
            pre_rows = run_sql_fetchall(precheck_sql)
        except Exception as exc:
            return _execution_failure(exc, component="precheck", failed_sql=precheck_sql)

    if sql_type == "select":
        if not is_safe_query_sql(sql):
            return _execution_failure(
                "조회 SQL 안전성 검사 실패",
                component="main",
                failed_sql=sql,
                precheck_result=pre_rows,
            )

        statements = split_sql_statements(sql)
        statement_results: list[dict[str, Any]] = []
        for index, statement_sql in enumerate(statements):
            try:
                rows = run_sql_fetchall(statement_sql) if can_use_live_db() else offline_select_rows(statement_sql)
            except Exception as exc:
                return _execution_failure(
                    exc,
                    component="main",
                    failed_sql=statement_sql,
                    statement_index=index,
                    statement_results=statement_results,
                    precheck_result=pre_rows,
                )
            statement_results.append(_statement_result(index, statement_sql, list(rows)))

        return {
            "sql_result": statement_results[-1]["rows"] if statement_results else None,
            "statement_results": statement_results,
            "row_count": statement_results[-1]["row_count"] if statement_results else 0,
            "precheck_result": pre_rows,
            "postcheck_result": None,
            "failed_sql_component": None,
            "failed_statement_index": None,
            "failed_statement_sql": "",
            "execution_error_info": {},
            "error": ""
        }

    ok, reason = is_safe_mart_sql(sql, target_table)
    if not ok:
        return _execution_failure(
            reason,
            component="main",
            failed_sql=sql,
            precheck_result=pre_rows,
        )

    if can_use_live_db():
        try:
            ensure_target_schema_exists(target_table)
            run_sql_commit(sql)
        except Exception as exc:
            error_text = str(exc)
            if sql_type == "create_table_as" and target_table and "already exists" in error_text.lower():
                try:
                    drop_table_if_exists(target_table)
                    run_sql_commit(sql)
                except Exception as retry_exc:
                    return _execution_failure(
                        retry_exc,
                        component="main",
                        failed_sql=sql,
                        precheck_result=pre_rows,
                    )
            else:
                return _execution_failure(
                    exc,
                    component="main",
                    failed_sql=sql,
                    precheck_result=pre_rows,
                )

    post_rows = None
    postcheck_sql = str(state["sql_draft"].get("postcheck_sql") or "").strip()
    if postcheck_sql and can_use_live_db():
        try:
            post_rows = run_sql_fetchall(postcheck_sql)
        except Exception as exc:
            return _execution_failure(
                exc,
                component="postcheck",
                failed_sql=postcheck_sql,
                precheck_result=pre_rows,
            )

    # 마트 실데이터 미리보기를 sql_result 로 반환한다. 하류는 마트를 DB에서 직접 조회한다.
    if can_use_live_db():
        try:
            mart_rows = run_sql_fetchall(f"SELECT * FROM {target_table} LIMIT {MART_PREVIEW_ROWS}")
        except Exception:
            mart_rows = [("마트 생성 완료", target_table)]
    else:
        mart_rows = offline_mart_rows(target_table)

    return {
        "sql_result": mart_rows,
        "statement_results": [],
        "row_count": len(mart_rows),
        "precheck_result": pre_rows,
        "postcheck_result": post_rows,
        "failed_sql_component": None,
        "failed_statement_index": None,
        "failed_statement_sql": "",
        "execution_error_info": {},
        "error": ""
    }

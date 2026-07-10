"""execute_sql 노드: 안전성 검사 후 SQL 실행(조회/마트)."""

from __future__ import annotations

import os

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

# 마트 생성 후 sql_result 에 담을 미리보기 행 수. 하류(EDA/분석)는 마트를 DB 로 직접
# 조회(전체)하므로 이 미리보기는 검증/최종답변 표시·CSV 폴백용 소량 샘플이다.
MART_PREVIEW_ROWS = int(os.getenv("MART_PREVIEW_ROWS", "100"))


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

        # 마트 실데이터 미리보기를 sql_result 로 반환한다. 과거엔 ("마트 생성 완료", table)
        # 확인메시지를 넣어 하류가 col_1/col_2 만 받아 분석에 실패했다. 실데이터 미리보기를
        # 넣으면 검증·최종답변 표시와 CSV 폴백이 실제 컬럼을 갖는다(전체 데이터는 하류가 DB 직접조회).
        if can_use_live_db():
            try:
                mart_rows = run_sql_fetchall(f"SELECT * FROM {target_table} LIMIT {MART_PREVIEW_ROWS}")
            except Exception:
                mart_rows = [("마트 생성 완료", target_table)]
        else:
            mart_rows = offline_mart_rows(target_table)

        return {
            "sql_result": mart_rows,
            "row_count": len(mart_rows),
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

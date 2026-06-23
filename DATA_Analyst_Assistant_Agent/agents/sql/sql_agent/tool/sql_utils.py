from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy import text

from DATA_Analyst_Assistant_Agent.agents.sql.self_check import mysql_dialect_error
from DATA_Analyst_Assistant_Agent.agents.sql.sql_agent.tool.runtime import (
    ALLOWED_MART_SCHEMA,
    ALLOW_MART_WRITE,
    engine,
)


def safe_json_parse(text_value: str, fallback: dict) -> dict:
    cleaned = text_value.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return fallback


def clean_sql(sql: str) -> str:
    sql = sql.replace("```sql", "").replace("```", "").strip()
    if not sql.endswith(";"):
        sql += ";"
    return sql


def format_result_rows(rows: Any, max_rows: int = 10) -> str:
    if not rows:
        return "결과 없음"
    preview = rows[:max_rows]
    return "\n".join([str(tuple(r)) for r in preview])


def is_safe_query_sql(sql: str) -> bool:
    lowered = sql.strip().lower()
    if lowered.startswith("select") or lowered.startswith("with"):
        banned = ["insert ", "update ", "delete ", "drop ", "alter ", "truncate ", "create "]
        return not any(k in lowered for k in banned)
    return False


def is_safe_mart_sql(sql: str, target_table: Optional[str]) -> tuple[bool, str]:
    lowered = sql.strip().lower()

    if not ALLOW_MART_WRITE:
        return False, "현재 설정상 마트 생성 SQL 실행이 비활성화되어 있습니다."

    banned = [
        "drop database", "drop schema", "truncate ", "alter table",
        "grant ", "revoke ", "rename table",
    ]
    if any(k in lowered for k in banned):
        return False, "위험한 DDL/DCL 문이 포함되어 있습니다."

    allowed_prefixes = ["create table", "create or replace table", "insert into"]
    if not any(lowered.startswith(p) for p in allowed_prefixes):
        return False, "허용되지 않은 마트 생성 SQL 형식입니다."

    if target_table:
        if ALLOWED_MART_SCHEMA.lower() not in target_table.lower():
            return False, f"타겟 테이블은 허용된 스키마({ALLOWED_MART_SCHEMA}) 안에 있어야 합니다."

    return True, ""


def run_sql_fetchall(sql: str):
    with engine.connect() as conn:
        return conn.execute(text(sql)).fetchall()


def run_sql_commit(sql: str):
    with engine.begin() as conn:
        conn.execute(text(sql))


def offline_select_rows(sql: str):
    return [("offline_dry_run", sql[:120])]


def offline_mart_rows(target_table: Optional[str]):
    return [("offline_dry_run", target_table or "target_table")]


def can_use_live_db() -> bool:
    return engine is not None


def validate_mysql_sql(sql: str) -> str:
    return mysql_dialect_error(sql)

"""SQL 에이전트 공용 런타임.

- 설정 상수 (MAX_RETRIES / ALLOWED_MART_SCHEMA / ALLOW_MART_WRITE)
- 엔진/LLM 메모이즈 접근자 (get_engine / get_llm)
- SQL 안전성·파싱·포맷 헬퍼

[동작 보존 핵심] 기존 모듈은 import 시 `engine = get_db_engine()`,
`llm = get_chat_model()` 를 즉시 생성했다. 이를 최초 호출 시 1회 생성하는
메모이즈 접근자로 바꿔 import 부작용(즉시 DB 연결/LLM 생성)을 제거한다.
"엔진 1개 / LLM 1개" 동작은 그대로 유지된다.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

from sqlalchemy import text

from DATA_Analyst_Assistant_Agent.shared.db import get_db_engine
from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model
from DATA_Analyst_Assistant_Agent.agents.sql.self_check import mysql_dialect_error
from DATA_Analyst_Assistant_Agent.agents.sql.sql_text import split_sql_statements
import DATA_Analyst_Assistant_Agent.shared.config  # noqa: F401  (.env 로드 + DB_*/MYSQL_* 별칭 정규화)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 2))
ALLOWED_MART_SCHEMA = os.getenv("ALLOWED_MART_SCHEMA", "analytics")
ALLOW_MART_WRITE = os.getenv("ALLOW_MART_WRITE", "true").lower() == "true"
MYSQL_DIALECT_NAME = "MySQL 8.x"


# -----------------------------
# 메모이즈 싱글톤 접근자
# -----------------------------
_engine = None
_llm = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = get_db_engine()
    return _engine


def get_llm():
    global _llm
    if _llm is None:
        _llm = get_chat_model(temperature=0)
    return _llm


# -----------------------------
# Utils
# -----------------------------
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
        "grant ", "revoke ", "rename table"
    ]
    if any(k in lowered for k in banned):
        return False, "위험한 DDL/DCL 문이 포함되어 있습니다."

    statements = split_sql_statements(sql)
    if len(statements) != 1:
        return False, "마트 생성 SQL은 단일 statement여야 합니다."
    if not re.match(
        r"^\s*CREATE\s+TABLE\s+.+?\s+AS\s+(?:WITH\b|SELECT\b)",
        statements[0],
        re.IGNORECASE | re.DOTALL,
    ):
        return False, "CREATE TABLE ... AS SELECT 형식만 허용됩니다."

    if target_table:
        target_table_lower = target_table.lower()
        if ALLOWED_MART_SCHEMA.lower() not in target_table_lower:
            return False, f"타겟 테이블은 허용된 스키마({ALLOWED_MART_SCHEMA}) 안에 있어야 합니다."

    return True, ""


def run_sql_fetchall(sql: str):
    with get_engine().connect() as conn:
        return conn.execute(text(sql)).fetchall()


def run_sql_commit(sql: str):
    with get_engine().begin() as conn:
        conn.execute(text(sql))


def extract_target_schema(target_table: Optional[str]) -> Optional[str]:
    if not target_table:
        return None
    normalized = target_table.strip().strip("`")
    if "." not in normalized:
        return None
    schema, _table = normalized.split(".", 1)
    schema = schema.strip().strip("`")
    return schema or None


def ensure_target_schema_exists(target_table: Optional[str]) -> Optional[str]:
    schema = extract_target_schema(target_table)
    if not schema:
        return None
    with get_engine().begin() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{schema}`"))
    return schema


def offline_select_rows(sql: str):
    return [("offline_dry_run", sql[:120])]


def offline_mart_rows(target_table: Optional[str]):
    return [("offline_dry_run", target_table or "target_table")]


def can_use_live_db() -> bool:
    try:
        return get_engine() is not None
    except Exception:
        return False


def validate_mysql_sql(sql: str) -> str:
    return mysql_dialect_error(sql)


def drop_table_if_exists(target_table: Optional[str]) -> None:
    if not target_table:
        return
    with get_engine().begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {target_table}"))


def infer_target_table_from_sql(sql: str) -> Optional[str]:
    match = re.search(r"(?is)create\s+(?:or\s+replace\s+)?table\s+([`\\w\\.]+)", sql or "")
    if match:
        return match.group(1).strip("`")
    match = re.search(r"(?is)insert\s+into\s+([`\\w\\.]+)", sql or "")
    if match:
        return match.group(1).strip("`")
    return None

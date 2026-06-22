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

import json
import os
from typing import Any, Optional

from sqlalchemy import text

from DATA_Analyst_Assistant_Agent.shared.db import get_db_engine
from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model
import DATA_Analyst_Assistant_Agent.shared.config  # noqa: F401  (.env 로드 + DB_*/MYSQL_* 별칭 정규화)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 2))
ALLOWED_MART_SCHEMA = os.getenv("ALLOWED_MART_SCHEMA", "analytics")
ALLOW_MART_WRITE = os.getenv("ALLOW_MART_WRITE", "true").lower() == "true"


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
        "grant ", "revoke ", "rename table"
    ]
    if any(k in lowered for k in banned):
        return False, "위험한 DDL/DCL 문이 포함되어 있습니다."

    allowed_prefixes = [
        "create table",
        "create or replace table",
        "insert into"
    ]
    if not any(lowered.startswith(p) for p in allowed_prefixes):
        return False, "허용되지 않은 마트 생성 SQL 형식입니다."

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

from __future__ import annotations

import re

import pymysql

# 사용자에게 보여줄 필요 없는 MySQL 내부 시스템 DB들
SYSTEM_DATABASES = {"information_schema", "mysql", "performance_schema", "sys"}

SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def quote_identifier(name: str) -> str:
    if not SAFE_IDENTIFIER.fullmatch(name):
        raise ValueError(f"허용되지 않는 이름입니다: {name}")
    return f"`{name}`"

def connect(host: str, port: int, user: str, password: str, database: str | None = None):
    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset="utf8mb4",
        connect_timeout=10,
        read_timeout=30,
    )

def list_databases(host: str, port: int, user: str, password: str) -> list[str]:
    conn = connect(host, port, user, password)
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW DATABASES")
            names = [row[0] for row in cur.fetchall()]
        return [n for n in names if n.lower() not in SYSTEM_DATABASES]
    finally:
        conn.close()

def list_tables(host: str, port: int, user: str, password: str, database: str) -> list[str]:
    quote_identifier(database)
    conn = connect(host, port, user, password, database=database)
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES")
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def preview_table(
    host: str, port: int, user: str, password: str,
    database: str, table: str, limit: int = 50,
) -> tuple[list[str], list[dict]]:
    """테이블 데이터를 (컬럼 목록, 행 목록)으로 돌려준다."""
    quote_identifier(database)
    quoted_table = quote_identifier(table)  # 검증 + 백틱
    conn = connect(host, port, user, password, database=database)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {quoted_table} LIMIT %s", (int(limit),))
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description or []]
        return columns, [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()
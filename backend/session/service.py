from __future__ import annotations

import re
from collections.abc import Iterator
from uuid import uuid4

from fastapi import HTTPException

from backend.config import StorageMySQL
from backend.mysql import db
from backend.session.schemas import SessionCreateRequest, SessionResponse
from backend.session.store import META_DATABASE, SESSIONS_TABLE, ensure_session_store, storage_connect
from backend.storage.ingest import ingest_database

# DB 이름으로 사용불가하기 때문에 특수문자가 들어갈경우 _로 변경 
def sanitize_username(username: str) -> str:
    sanitized = re.sub(r"\W+", "_", username).strip("_").lower()
    return sanitized or "user"

# 세션 id 생성
def new_session_id() -> str:
    return uuid4().hex[:8]

# 정해진 로직의 DB이름 생성
def build_database_names(username: str, session_id: str) -> tuple[str, str]:
    safe_user = sanitize_username(username)
    base = f"s_{safe_user}_{session_id}"

    if len(base) > 64:
        max_user_len = 64 - len("s__") - len(session_id)
        safe_user = safe_user[:max_user_len]
        base = f"s_{safe_user}_{session_id}"

    mart = f"{base}__mart"
    if len(mart) > 64:
        max_base_len = 64 - len("__mart")
        base = base[:max_base_len]
        mart = f"{base}__mart"

    db.quote_identifier(base)
    db.quote_identifier(mart)
    return base, mart


# DB에서 가져온 row 값(dict)들을 Pydantic 모델로 변환
def row_to_session(row: dict) -> SessionResponse:
    return SessionResponse(
        id=row["id"],
        username=row["username"],
        title=row["title"],
        source_host=row["source_host"],
        source_port=row["source_port"],
        source_user=row["source_user"],
        source_database=row["source_database"],
        session_db=row["session_db"],
        mart_db=row["mart_db"],
        status=row["status"],
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        last_opened_at=row.get("last_opened_at"),
    )


# 세션의 데이터마트 DB 생성
def create_mart_database(mart_db: str) -> None:
    quoted = db.quote_identifier(mart_db)
    conn = storage_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS {quoted} CHARACTER SET utf8mb4")
        conn.commit()
    finally:
        conn.close()


# 데이터 마트 DB만 지우고 세션 DB는 지우지 않음
def reset_mart_database(mart_db: str) -> None:
    quoted = db.quote_identifier(mart_db)
    conn = storage_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {quoted}")
            cur.execute(f"CREATE DATABASE {quoted} CHARACTER SET utf8mb4")
        conn.commit()
    finally:
        conn.close()

# 사용자 세션 DB 생성
def create_session(username: str, payload: SessionCreateRequest) -> tuple[SessionResponse, dict[str, int]]:
    ensure_session_store()

    session_id = new_session_id()
    session_db, mart_db = build_database_names(username, session_id)
    title = payload.title or payload.database

    # 사용자의 원격 DB를 사용자의 세션 DB로 복사
    copied = ingest_database(
        payload.host,
        payload.port,
        payload.user,
        payload.password,
        payload.database,
        target_database=session_db,
    )

    create_mart_database(mart_db)

    conn = storage_connect(META_DATABASE)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO `{SESSIONS_TABLE}` (
                    id,
                    username,
                    title,
                    source_host,
                    source_port,
                    source_user,
                    source_database,
                    session_db,
                    mart_db,
                    status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    username,
                    title,
                    payload.host,
                    payload.port,
                    payload.user,
                    payload.database,
                    session_db,
                    mart_db,
                    "ready",
                ),
            )
        conn.commit()
    finally:
        conn.close()

    session = get_owned_session(session_id, username)
    return session, copied


# 사용자 자신의 세션 목록 반환
def list_sessions(username: str) -> list[SessionResponse]:
    ensure_session_store()

    conn = storage_connect(META_DATABASE)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    id,
                    username,
                    title,
                    source_host,
                    source_port,
                    source_user,
                    source_database,
                    session_db,
                    mart_db,
                    status,
                    created_at,
                    updated_at,
                    last_opened_at
                FROM `{SESSIONS_TABLE}`
                WHERE username = %s
                ORDER BY COALESCE(last_opened_at, created_at) DESC
                """,
                (username,),
            )
            columns = [desc[0] for desc in cur.description or []]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    finally:
        conn.close()

    return [row_to_session(row) for row in rows]

# 사용자 자신의 세션 조회
def get_owned_session(session_id: str, username: str) -> SessionResponse:
    ensure_session_store()

    conn = storage_connect(META_DATABASE)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    id,
                    username,
                    title,
                    source_host,
                    source_port,
                    source_user,
                    source_database,
                    session_db,
                    mart_db,
                    status,
                    created_at,
                    updated_at,
                    last_opened_at
                FROM `{SESSIONS_TABLE}`
                WHERE id = %s AND username = %s
                """,
                (session_id, username),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

            columns = [desc[0] for desc in cur.description or []]
            session = row_to_session(dict(zip(columns, row)))

            cur.execute(
                f"""
                UPDATE `{SESSIONS_TABLE}`
                SET last_opened_at = CURRENT_TIMESTAMP
                WHERE id = %s AND username = %s
                """,
                (session_id, username),
            )
        conn.commit()
    finally:
        conn.close()

    return session

# 사용자 세션의 테이블 목록 반환 (원격 DB 복사본)
def list_session_tables(session_id: str, username: str) -> list[str]:
    session = get_owned_session(session_id, username)
    return db.list_tables(
        StorageMySQL.HOST,
        StorageMySQL.PORT,
        StorageMySQL.USER,
        StorageMySQL.PASSWORD,
        session.session_db,
    )

# 사용자 세션의 테이블 데이터 반환
def preview_session_table(
    session_id: str,
    username: str,
    table: str,
    limit: int = 50,
    sort_by: str | None = None,
    sort_order: str = "asc",
    cursor: str | None = None,
) -> db.TablePreviewPage:
    session = get_owned_session(session_id, username)
    return db.preview_table_page(
        StorageMySQL.HOST,
        StorageMySQL.PORT,
        StorageMySQL.USER,
        StorageMySQL.PASSWORD,
        session.session_db,
        table,
        limit,
        sort_by,
        sort_order,
        cursor,
    )


def export_session_table_csv(
    session_id: str,
    username: str,
    table: str,
    sort_by: str | None = None,
    sort_order: str = "asc",
) -> Iterator[str]:
    session = get_owned_session(session_id, username)
    return db.stream_table_csv(
        StorageMySQL.HOST,
        StorageMySQL.PORT,
        StorageMySQL.USER,
        StorageMySQL.PASSWORD,
        session.session_db,
        table,
        sort_by,
        sort_order,
    )

# 사용자 id를 받고 reset_mart_database함수 실행하여 데이터 마트 삭제
def reset_session_mart(session_id: str, username: str) -> SessionResponse:
    session = get_owned_session(session_id, username)
    reset_mart_database(session.mart_db)
    return get_owned_session(session_id, username)

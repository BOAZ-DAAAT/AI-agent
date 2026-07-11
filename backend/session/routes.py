from __future__ import annotations

import pymysql
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.auth.deps import get_current_user
from backend.session.schemas import (
    PreviewResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionListResponse,
    SessionResponse,
    TablesResponse,
)
from backend.session.service import (
    create_session,
    get_owned_session,
    list_session_tables,
    list_sessions,
    preview_session_table,
    reset_session_mart,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])

# 로그인 토큰에서 사용자 이름(sub) 추출
def current_username(user: dict = Depends(get_current_user)) -> str:
    return str(user["sub"])

# 사용자가 원격 DB를 선택할 경우 호출
@router.post("", response_model=SessionCreateResponse)
def create_session_route(
    payload: SessionCreateRequest,
    username: str = Depends(current_username),
) -> SessionCreateResponse:
    try:
        # 세션 생성 복사 로직
        session, tables = create_session(username, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except pymysql.MySQLError as exc:
        raise HTTPException(status_code=502, detail=f"세션 생성 실패: {exc}")
    return SessionCreateResponse(session=session, tables=tables)

# 현재 사용자 세션 목록 반환
@router.get("", response_model=SessionListResponse)
def list_sessions_route(username: str = Depends(current_username)) -> SessionListResponse:
    try:
        sessions = list_sessions(username)
    except pymysql.MySQLError as exc:
        raise HTTPException(status_code=502, detail=f"세션 목록 조회 실패: {exc}")
    return SessionListResponse(sessions=sessions)

# 사용자가 선택한 세션 조회
@router.get("/{session_id}", response_model=SessionResponse)
def get_session_route(
    session_id: str,
    username: str = Depends(current_username),
) -> SessionResponse:
    try:
        return get_owned_session(session_id, username)
    except pymysql.MySQLError as exc:
        raise HTTPException(status_code=502, detail=f"세션 조회 실패: {exc}")

# 사용자 세션의 원본 DB 테이블 목록 반환
@router.get("/{session_id}/tables", response_model=TablesResponse)
def list_session_tables_route(
    session_id: str,
    username: str = Depends(current_username),
) -> TablesResponse:
    try:
        tables = list_session_tables(session_id, username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except pymysql.MySQLError as exc:
        raise HTTPException(status_code=502, detail=f"테이블 목록 조회 실패: {exc}")
    return TablesResponse(tables=tables)

# 사용자 세션의 원본 DB 테이블 샘플 반환
@router.get("/{session_id}/tables/{table}/preview", response_model=PreviewResponse)
def preview_session_table_route(
    session_id: str,
    table: str,
    limit: int = Query(default=50, ge=1, le=1000),
    username: str = Depends(current_username),
) -> PreviewResponse:
    try:
        columns, rows = preview_session_table(session_id, username, table, limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except pymysql.MySQLError as exc:
        raise HTTPException(status_code=502, detail=f"테이블 미리보기 실패: {exc}")
    return PreviewResponse(columns=columns, rows=rows)

# 사용자 세션의 마트 DB 초기화
@router.post("/{session_id}/reset", response_model=SessionResponse)
def reset_session_mart_route(
    session_id: str,
    username: str = Depends(current_username),
) -> SessionResponse:
    try:
        return reset_session_mart(session_id, username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except pymysql.MySQLError as exc:
        raise HTTPException(status_code=502, detail=f"마트 초기화 실패: {exc}")
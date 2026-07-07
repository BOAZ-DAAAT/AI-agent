from __future__ import annotations

import pymysql
from fastapi import APIRouter, HTTPException

from backend.mysql import db
from backend.mysql.schemas import (
    DatabasesRequest,
    DatabasesResponse,
    PreviewRequest,
    PreviewResponse,
    TablesRequest,
    TablesResponse,
)

router = APIRouter(prefix="/mysql", tags=["mysql"])

#원격 MySQL 서버의 DB 목록을 돌려준다.
@router.post("/databases", response_model=DatabasesResponse)
def list_databases(payload: DatabasesRequest) -> DatabasesResponse:
    
    try:
        names = db.list_databases(payload.host, payload.port, payload.user, payload.password)
    except pymysql.MySQLError as exc:
        raise HTTPException(status_code=502, detail=f"MySQL 연결/조회 실패: {exc}")
    return DatabasesResponse(databases=names)

#선택한 DB 안의 테이블 목록을 돌려준다.
@router.post("/tables", response_model=TablesResponse)
def list_tables(payload: TablesRequest) -> TablesResponse:
    try:
        names = db.list_tables(
            payload.host, payload.port, payload.user, payload.password,
            database=payload.database,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except pymysql.MySQLError as exc:
        raise HTTPException(status_code=502, detail=f"MySQL 연결/조회 실패: {exc}")
    return TablesResponse(tables=names)

#테이블 데이터를 미리보기로 돌려준다.
@router.post("/preview", response_model=PreviewResponse)
def preview_table(payload: PreviewRequest) -> PreviewResponse:
    try:
        columns, rows = db.preview_table(
            payload.host, payload.port, payload.user, payload.password,
            database=payload.database, table=payload.table, limit=payload.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except pymysql.MySQLError as exc:
        raise HTTPException(status_code=502, detail=f"MySQL 연결/조회 실패: {exc}")
    return PreviewResponse(columns=columns, rows=rows)
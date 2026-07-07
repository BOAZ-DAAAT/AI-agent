from __future__ import annotations

import pymysql
from fastapi import APIRouter, HTTPException, Query

from backend.config import StorageMySQL
from backend.mysql import db
from backend.mysql.schemas import DatabasesResponse, PreviewResponse, TablesResponse
from backend.storage.ingest import ingest_database
from backend.storage.schemas import IngestRequest, IngestResponse

router = APIRouter(prefix="/storage", tags=["storage"])

# 사용자 원격 DB의 모든 테이블을 로컬 저장소로 복사
@router.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
    try:
        copied = ingest_database(
            payload.host, payload.port, payload.user, payload.password,
            payload.database, target_database=payload.target_database,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except pymysql.MySQLError as exc:
        raise HTTPException(status_code=502, detail=f"적재 실패: {exc}")
    return IngestResponse(
        target_database=payload.target_database or payload.database,
        tables=copied,
    )

# 로컬(우리 서버) 저장소에 있는 DB 목록 조회
@router.get("/databases", response_model=DatabasesResponse)
def list_local_databases() -> DatabasesResponse:
    try:
        names = db.list_databases(
            StorageMySQL.HOST, StorageMySQL.PORT, StorageMySQL.USER, StorageMySQL.PASSWORD,
        )
    except pymysql.MySQLError as exc:
        raise HTTPException(status_code=502, detail=f"저장소 조회 실패: {exc}")
    return DatabasesResponse(databases=names)

# ingest된 사본 DB의 테이블 목록
@router.get("/{database}/tables", response_model=TablesResponse)
def list_local_tables(database: str) -> TablesResponse:
    try:
        names = db.list_tables(
            StorageMySQL.HOST, StorageMySQL.PORT, StorageMySQL.USER, StorageMySQL.PASSWORD,
            database=database,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except pymysql.MySQLError as exc:
        raise HTTPException(status_code=502, detail=f"저장소 조회 실패: {exc}")
    return TablesResponse(tables=names)

# ingest된 테이블 데이터 미리보기
@router.get("/{database}/tables/{table}/preview", response_model=PreviewResponse)
def preview_local_table(
    database: str,
    table: str,
    limit: int = Query(default=50, ge=1, le=1000),
) -> PreviewResponse:
    try:
        columns, rows = db.preview_table(
            StorageMySQL.HOST, StorageMySQL.PORT, StorageMySQL.USER, StorageMySQL.PASSWORD,
            database=database, table=table, limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except pymysql.MySQLError as exc:
        raise HTTPException(status_code=502, detail=f"저장소 조회 실패: {exc}")
    return PreviewResponse(columns=columns, rows=rows)
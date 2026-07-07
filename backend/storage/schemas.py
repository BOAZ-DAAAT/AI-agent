from __future__ import annotations

from pydantic import BaseModel

from backend.mysql.schemas import MySQLConn


# ingest 요청: 원격 접속정보 + 원본 DB 이름 + 
class IngestRequest(MySQLConn):
    database: str
    target_database: str | None = None


# ingest 결과: 사본 DB 이름 + 테이블별 복사 행수
class IngestResponse(BaseModel):
    target_database: str
    tables: dict[str, int]


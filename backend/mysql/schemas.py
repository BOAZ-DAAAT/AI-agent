from __future__ import annotations

from pydantic import BaseModel, Field

# 요청

class MySQLConn(BaseModel):
    """원격 MySQL 접속정보. 서버에 저장하지 않으므로 매 요청에 담아 보낸다."""
    host: str
    port: int = 3306
    user: str
    password: str


class DatabasesRequest(MySQLConn):
    """DB 목록 조회 — 접속정보만 있으면 됨."""


class TablesRequest(MySQLConn):
    """테이블 목록 조회 — 대상 DB 지정."""
    database: str


class PreviewRequest(MySQLConn):
    """테이블 데이터 미리보기."""
    database: str
    table: str
    limit: int = Field(default=50, ge=1, le=1000)


# 응답

class DatabasesResponse(BaseModel):
    databases: list[str]


class TablesResponse(BaseModel):
    tables: list[str]


class PreviewResponse(BaseModel):
    columns: list[str]
    rows: list[dict]
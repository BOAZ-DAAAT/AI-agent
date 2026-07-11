from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from backend.mysql.schemas import MySQLConn


class SessionCreateRequest(MySQLConn):
    database: str
    title: str | None = None


class SessionResponse(BaseModel):
    id: str
    username: str
    title: str
    source_host: str
    source_port: int
    source_user: str
    source_database: str
    session_db: str
    mart_db: str
    status: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_opened_at: datetime | None = None


class SessionCreateResponse(BaseModel):
    session: SessionResponse
    tables: dict[str, int]


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]


class TablesResponse(BaseModel):
    tables: list[str]


class PreviewResponse(BaseModel):
    columns: list[str]
    rows: list[dict]
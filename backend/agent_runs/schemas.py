from __future__ import annotations

from pydantic import BaseModel


class AgentRunCreateRequest(BaseModel):
    session_id: str
    query: str


class AgentRunResponse(BaseModel):
    run_id: str
    thread_id: str
    status: str
    query: str
    session_id: str

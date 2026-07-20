from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator


class AgentRunCreateRequest(BaseModel):
    session_id: str
    query: str


class AgentRunResponse(BaseModel):
    run_id: str
    thread_id: str
    status: str
    query: str
    session_id: str


class AgentRunResumeRequest(BaseModel):
    type: Literal["clarification"]
    answer: str

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        answer = value.strip()
        if not answer:
            raise ValueError("답변을 입력해주세요.")
        return answer


class AgentRunResumeResponse(BaseModel):
    run_id: str
    thread_id: str
    status: Literal["running"]
    resume_type: Literal["clarification"]

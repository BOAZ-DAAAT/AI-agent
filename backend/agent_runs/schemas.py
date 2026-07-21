from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

from DATA_Analyst_Assistant_Agent.supervisor.summary.schemas import NodeSummaryResult


class AgentRunCreateRequest(BaseModel):
    session_id: str
    query: str


class AgentRunResponse(BaseModel):
    run_id: str
    thread_id: str
    status: str
    query: str
    session_id: str


class AgentNodeSummaryResponse(BaseModel):
    run_id: str
    node_id: str
    agent_name: str
    summary_artifact_id: str
    summary: NodeSummaryResult


class AgentRunDeleteResponse(BaseModel):
    run_id: str
    deleted_event_count: int
    deleted_artifact_count: int


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


BranchStage = Literal["sql", "eda", "analysis", "insight"]


class AgentRunBranchRequest(BaseModel):
    start_stage: BranchStage
    instruction: str
    parent_node_id: str | None = None

    @field_validator("instruction")
    @classmethod
    def validate_instruction(cls, value: str) -> str:
        instruction = value.strip()
        if not instruction:
            raise ValueError("분기 지시사항을 입력해주세요.")
        return instruction


class AgentRunBranchResponse(BaseModel):
    run_id: str
    thread_id: str
    status: Literal["created"]
    start_stage: BranchStage
    source_run_id: str

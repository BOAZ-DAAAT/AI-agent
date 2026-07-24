from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

from DATA_Analyst_Assistant_Agent.supervisor.summary.schemas import NodeSummaryResult
from DATA_Analyst_Assistant_Agent.supervisor.report.schemas import ReportResult


class AgentRunCreateRequest(BaseModel):
    session_id: str
    query: str


class AgentRunResponse(BaseModel):
    run_id: str
    thread_id: str
    status: str
    query: str
    session_id: str


ResumeType = Literal["clarification", "analysis_review", "approval"]


class AgentNodeSummaryResponse(BaseModel):
    run_id: str
    node_id: str
    agent_name: str
    summary_artifact_id: str
    summary: NodeSummaryResult


class AgentNodeReportResponse(BaseModel):
    run_id: str
    node_id: str
    report_artifact_id: str
    created_at: str
    report: ReportResult


class AgentReportListItem(BaseModel):
    run_id: str
    report_artifact_id: str
    created_at: str
    report: ReportResult


class AgentReportListResponse(BaseModel):
    reports: list[AgentReportListItem]


class AgentRunDeleteResponse(BaseModel):
    run_id: str
    deleted_event_count: int
    deleted_artifact_count: int


class AgentSessionRunsDeleteResponse(BaseModel):
    session_id: str
    deleted_run_count: int
    deleted_event_count: int
    deleted_artifact_count: int


class AgentRunCancelResponse(BaseModel):
    run_id: str
    status: Literal["cancelled"]
    discarded_node_id: str | None = None


class AgentRunResumeRequest(BaseModel):
    """세 가지 재개 유형을 하나로 받는다 — type이 어느 필드가 필요한지 결정한다.

    - clarification: answer
    - analysis_review: approval_id + (selected_option_id 또는 free_text 중 하나)
    - approval: approved(true/false 모두 허용) + reason(거부 시 선택)
    """

    type: ResumeType
    answer: str | None = None
    approval_id: str | None = None
    selected_option_id: str | None = None
    free_text: str | None = None
    approved: bool | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_fields_for_type(self) -> "AgentRunResumeRequest":
        if self.type == "clarification":
            answer = (self.answer or "").strip()
            if not answer:
                raise ValueError("답변을 입력해주세요.")
            self.answer = answer
        elif self.type == "analysis_review":
            approval_id = (self.approval_id or "").strip()
            if not approval_id:
                raise ValueError("approval_id가 필요합니다.")
            self.approval_id = approval_id
            selected_option_id = (self.selected_option_id or "").strip() or None
            free_text = (self.free_text or "").strip() or None
            self.selected_option_id = selected_option_id
            self.free_text = free_text
            if (selected_option_id is None) == (free_text is None):
                raise ValueError("selected_option_id 또는 free_text 중 정확히 하나를 입력해주세요.")
        elif self.type == "approval":
            if not isinstance(self.approved, bool):
                raise ValueError("approved는 boolean이어야 합니다.")
            self.reason = (self.reason or "").strip() or None
        return self


class AgentRunResumeResponse(BaseModel):
    run_id: str
    thread_id: str
    status: Literal["running"]
    resume_type: ResumeType


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

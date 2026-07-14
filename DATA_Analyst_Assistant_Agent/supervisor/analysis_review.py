from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisSelectionResponse,
    ReviewOption,
    ReviewRequest,
)
from DATA_Analyst_Assistant_Agent.supervisor.state import AgentCompactResult


class InvalidAnalysisReviewRequest(ValueError):
    """Analysis 산출물의 구조화된 review 계약이 손상된 경우 발생한다."""


class AnalysisReviewResumePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_id: str
    selected_option_id: str | None = None
    free_text: str | None = None

    @field_validator("approval_id", "selected_option_id", "free_text", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        return value.strip()

    @model_validator(mode="after")
    def validate_payload(self) -> "AnalysisReviewResumePayload":
        if not self.approval_id:
            raise ValueError("approval_id는 비어 있을 수 없습니다.")
        AnalysisSelectionResponse(
            selected_option_id=self.selected_option_id,
            free_text=self.free_text,
        )
        return self

    def selection_response(self) -> AnalysisSelectionResponse:
        return AnalysisSelectionResponse(
            selected_option_id=self.selected_option_id,
            free_text=self.free_text,
        )


class AnalysisReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_id: str
    candidate_id: str
    review_request: ReviewRequest
    selection_response: AnalysisSelectionResponse
    selected_option: ReviewOption | None = None


def extract_analysis_review_request(result: AgentCompactResult) -> ReviewRequest | None:
    """Analysis result preview에서 review를 추출하며 손상된 계약은 fail-closed 처리한다."""

    if result.agent != "analysis_agent":
        return None
    found = False
    for artifact in result.artifacts:
        kind = str(artifact.metadata.get("kind") or artifact.kind or "")
        if kind != "analysis_result" or "review_request" not in artifact.preview:
            continue
        found = True
        value = artifact.preview.get("review_request")
        if value is None:
            return None
        try:
            return ReviewRequest.model_validate(value)
        except Exception as exc:
            raise InvalidAnalysisReviewRequest(
                f"invalid_analysis_review_request: {exc}"
            ) from exc
    return None if not found else None


def validate_analysis_review_resume(
    payload: dict[str, Any] | AnalysisReviewResumePayload,
    *,
    pending_approval: dict[str, Any],
    pending_candidate_id: str,
    review_request: ReviewRequest | dict[str, Any],
) -> AnalysisReviewDecision:
    """활성 승인과 사용자 선택을 교차 검증하고 정규화된 결정을 반환한다."""

    normalized = (
        payload
        if isinstance(payload, AnalysisReviewResumePayload)
        else AnalysisReviewResumePayload.model_validate(payload)
    )
    expected_approval_id = str(pending_approval.get("approval_id") or "")
    if normalized.approval_id != expected_approval_id:
        raise ValueError("analysis review approval_id가 활성 승인과 일치하지 않습니다.")

    expected_candidate_id = str(pending_approval.get("candidate_id") or "")
    if not expected_candidate_id or pending_candidate_id != expected_candidate_id:
        raise ValueError("analysis review candidate가 활성 후보와 일치하지 않습니다.")

    request = (
        review_request
        if isinstance(review_request, ReviewRequest)
        else ReviewRequest.model_validate(review_request)
    )
    selection = normalized.selection_response()
    selected_option = None
    if selection.selected_option_id:
        selected_option = next(
            (option for option in request.options if option.id == selection.selected_option_id),
            None,
        )
        if selected_option is None:
            raise ValueError("analysis review option이 review request에 존재하지 않습니다.")
    elif selection.free_text and not request.allow_free_text:
        raise ValueError("analysis review에서는 free text 입력이 허용되지 않습니다.")

    return AnalysisReviewDecision(
        approval_id=normalized.approval_id,
        candidate_id=pending_candidate_id,
        review_request=request,
        selection_response=selection,
        selected_option=selected_option,
    )

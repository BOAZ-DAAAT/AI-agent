from __future__ import annotations

from .common import BackendModel, JsonDict, StrEnum


class ApprovalStatus(StrEnum):
    pending = "pending"
    approved = "approved"
    edited = "edited"
    rejected = "rejected"
    expired = "expired"


class ApprovalDecision(StrEnum):
    approve = "approve"
    edit = "edit"
    reject = "reject"
    expire = "expire"


class ApprovalRequest(BackendModel):
    approval_id: str
    action: str
    resource: str
    payload: JsonDict
    status: ApprovalStatus
    created_at: str
    updated_at: str
    requested_by: str | None = None
    decided_by: str | None = None
    edited_payload: JsonDict | None = None
    reason: str | None = None
    run_id: str | None = None
    thread_id: str | None = None
    project_id: str | None = None


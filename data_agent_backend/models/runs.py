from __future__ import annotations

from .approvals import ApprovalRequest
from .artifacts import ArtifactRecord
from .common import BackendModel, JsonDict, StrEnum


class RunStatus(StrEnum):
    created = "created"
    running = "running"
    waiting_approval = "waiting_approval"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


TERMINAL_RUN_STATUSES = {RunStatus.succeeded, RunStatus.failed, RunStatus.cancelled}


class RunRecord(BackendModel):
    run_id: str
    thread_id: str | None = None
    project_id: str | None = None
    status: RunStatus = RunStatus.created
    metadata: JsonDict = {}
    created_at: str
    updated_at: str


class RunEvent(BackendModel):
    event_id: str
    run_id: str
    event_type: str
    message: str
    node_name: str | None = None
    tool_name: str | None = None
    artifact_ids: list[str] = []
    approval_id: str | None = None
    memory_ids: list[str] = []
    metadata: JsonDict = {}
    created_at: str


class RunSummary(BackendModel):
    run: RunRecord
    events: list[RunEvent] = []
    artifacts: list[ArtifactRecord] = []
    pending_approvals: list[ApprovalRequest] = []

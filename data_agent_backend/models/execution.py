from __future__ import annotations

from .artifacts import ArtifactRef
from .common import BackendModel, StrEnum


class ExecutionStatus(StrEnum):
    success = "success"
    error = "error"
    timeout = "timeout"
    policy_blocked = "policy_blocked"
    approval_required = "approval_required"
    sandbox_not_configured = "sandbox_not_configured"


class ExecutionLimits(BackendModel):
    timeout_ms: int = 30_000
    row_limit: int | None = None
    memory_mb: int | None = None


class ExecutionResult(BackendModel):
    execution_id: str
    status: ExecutionStatus
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    runtime_ms: int = 0
    created_artifact_ids: list[str] = []
    approval_id: str | None = None
    error_message: str | None = None


class PythonExecutionRequest(BackendModel):
    code: str
    inputs: list[ArtifactRef] = []
    limits: ExecutionLimits = ExecutionLimits()


from __future__ import annotations

from .common import BackendModel, JsonDict, StrEnum


class IntegrityStatus(StrEnum):
    pass_ = "pass"
    warning = "warning"
    fail = "fail"
    stale = "stale"
    pending = "pending"
    running = "running"


class IntegrityJobStatus(StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class IntegrityCheckResult(BackendModel):
    table_name: str | None = None
    status: IntegrityStatus
    severity: str = "info"
    summary: JsonDict = {}
    artifact_refs: list[JsonDict] = []
    metadata: JsonDict = {}

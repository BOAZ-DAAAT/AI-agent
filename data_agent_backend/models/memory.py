from __future__ import annotations

from .common import BackendModel, JsonDict, StrEnum


class MemoryType(StrEnum):
    user_preference = "user_preference"
    metric_definition = "metric_definition"
    business_glossary = "business_glossary"
    table_note = "table_note"
    column_note = "column_note"
    report_style = "report_style"
    analysis_pattern = "analysis_pattern"


class MemoryStatus(StrEnum):
    pending = "pending"
    active = "active"
    rejected = "rejected"
    archived = "archived"


class MemoryRecord(BackendModel):
    memory_id: str
    namespace: list[str]
    type: MemoryType
    status: MemoryStatus
    content_text: str
    content_json: JsonDict | None = None
    source: JsonDict = {}
    metadata: JsonDict = {}
    approval_id: str | None = None
    risk_flags: list[str] = []
    created_at: str
    updated_at: str


from __future__ import annotations

from .common import BackendModel, StrEnum


class RiskLevel(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"
    blocked = "blocked"


class PolicyDecision(BackendModel):
    decision_id: str
    allowed: bool
    requires_approval: bool
    risk_level: RiskLevel
    reason: str
    editable_fields: list[str] = []


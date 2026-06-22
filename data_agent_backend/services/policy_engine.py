from __future__ import annotations

from typing import Any

from data_agent_backend.models.common import BackendError, JsonDict, utc_now_iso
from data_agent_backend.models.contexts import PolicyContext
from data_agent_backend.models.ids import UUID4IdGenerator
from data_agent_backend.models.policy import PolicyDecision, RiskLevel
from data_agent_backend.storage.sqlite import SQLiteStore


class PolicyEngine:
    def __init__(self, sqlite: SQLiteStore, id_generator: UUID4IdGenerator | None = None) -> None:
        self.sqlite = sqlite
        self.id_generator = id_generator or UUID4IdGenerator()

    def evaluate(
        self,
        action: str,
        resource: str = "",
        payload: JsonDict | None = None,
        context: PolicyContext | None = None,
    ) -> PolicyDecision:
        context = context or PolicyContext()
        payload = payload or {}
        decision = self._decide(action, resource, payload, context)
        if not decision.allowed or decision.requires_approval:
            self.sqlite.insert_policy_audit(
                {
                    "decision_id": decision.decision_id,
                    "action": action,
                    "resource": resource,
                    "payload": payload,
                    "allowed": decision.allowed,
                    "requires_approval": decision.requires_approval,
                    "risk_level": decision.risk_level.value,
                    "reason": decision.reason,
                    "context": context.model_dump(mode="json"),
                    "created_at": utc_now_iso(),
                }
            )
        return decision

    def enforce(
        self,
        action: str,
        resource: str = "",
        payload: JsonDict | None = None,
        context: PolicyContext | None = None,
    ) -> PolicyDecision:
        decision = self.evaluate(action, resource, payload, context)
        if decision.requires_approval:
            raise BackendError(
                "APPROVAL_REQUIRED",
                decision.reason,
                {"decision_id": decision.decision_id, "editable_fields": decision.editable_fields},
            )
        if not decision.allowed:
            raise BackendError("POLICY_BLOCKED", decision.reason, {"decision_id": decision.decision_id, "resource": resource})
        return decision

    def _decision(
        self,
        allowed: bool,
        requires_approval: bool,
        risk_level: RiskLevel,
        reason: str,
        editable_fields: list[str] | None = None,
    ) -> PolicyDecision:
        return PolicyDecision(
            decision_id=self.id_generator.new_id("pd"),
            allowed=allowed,
            requires_approval=requires_approval,
            risk_level=risk_level,
            reason=reason,
            editable_fields=editable_fields or [],
        )

    def _decide(self, action: str, resource: str, payload: JsonDict, context: PolicyContext) -> PolicyDecision:
        resource_lower = resource.lower()
        if "/secrets" in resource_lower or resource_lower == "secrets":
            return self._decision(False, False, RiskLevel.blocked, "Secret access is blocked.")
        if action in {
            "catalog.read",
            "workspace.list",
            "workspace.read",
            "workspace.write",
            "artifact.preview",
            "artifact.list",
            "artifact.register",
            "memory.propose",
            "memory.read",
            "approval.read",
            "approval.resolve",
            "policy.evaluate",
        } or action.startswith("run."):
            if action == "workspace.write":
                if resource_lower.startswith("/artifacts"):
                    return self._decision(False, False, RiskLevel.blocked, "Direct write to /artifacts is not allowed.")
                if resource_lower.startswith("/catalog") or resource_lower.startswith("/skills"):
                    return self._decision(False, False, RiskLevel.blocked, "This mount is read-only.")
                if resource_lower.startswith("/memory"):
                    return self._decision(False, True, RiskLevel.medium, "Direct memory writes require approval.", ["content"])
                if resource_lower.startswith("/exports"):
                    return self._decision(False, True, RiskLevel.medium, "Exports must be created through an approved export tool.", ["destination"])
            return self._decision(True, False, RiskLevel.low, "Allowed by default MVP policy.")

        if action == "sql.run":
            if payload.get("blocked"):
                return self._decision(False, False, RiskLevel.blocked, str(payload.get("reason") or "SQL query is blocked."))
            if payload.get("row_limit", 0) > payload.get("max_row_limit", 10_000):
                return self._decision(False, True, RiskLevel.high, "SQL row limit is expensive and requires approval.", ["row_limit"])
            return self._decision(True, False, RiskLevel.low, "Read-only SQL is allowed.")

        if action in {"sandbox.python.run", "export.create"}:
            if context.approval_id:
                return self._decision(True, False, RiskLevel.medium, "Approved operation may proceed.")
            return self._decision(False, True, RiskLevel.high, f"{action} requires approval.", ["payload"])

        if action in {"shell.run", "network.call", "package.install"}:
            return self._decision(False, False, RiskLevel.blocked, f"{action} is blocked in MVP.")

        return self._decision(False, False, RiskLevel.blocked, f"Unknown or unsupported action: {action}.")


def decision_to_error(decision: PolicyDecision) -> BackendError:
    if decision.requires_approval:
        return BackendError("APPROVAL_REQUIRED", decision.reason, {"decision_id": decision.decision_id})
    return BackendError("POLICY_BLOCKED", decision.reason, {"decision_id": decision.decision_id})

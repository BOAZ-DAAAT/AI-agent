from __future__ import annotations

from data_agent_backend.models.approvals import ApprovalDecision, ApprovalRequest, ApprovalStatus
from data_agent_backend.models.common import BackendError, JsonDict, utc_now_iso
from data_agent_backend.models.contexts import PolicyContext
from data_agent_backend.models.ids import UUID4IdGenerator
from data_agent_backend.storage.sqlite import SQLiteStore, dumps_json, loads_json


SENSITIVE_KEYS = {"secret", "token", "password", "api_key", "apikey", "credential"}


class ApprovalStore:
    def __init__(self, sqlite: SQLiteStore, id_generator: UUID4IdGenerator | None = None) -> None:
        self.sqlite = sqlite
        self.id_generator = id_generator or UUID4IdGenerator()

    def create_approval_request(
        self,
        action: str,
        resource: str,
        payload: JsonDict,
        context: PolicyContext | None = None,
        requested_by: str | None = None,
    ) -> ApprovalRequest:
        context = context or PolicyContext()
        now = utc_now_iso()
        approval_id = self.id_generator.new_id("appr")
        redacted = self._redact(payload)
        self.sqlite.execute(
            """
            INSERT INTO approval_requests(
                approval_id, action, resource, payload_json, status, requested_by,
                run_id, thread_id, project_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                approval_id,
                action,
                resource,
                dumps_json(redacted),
                ApprovalStatus.pending.value,
                requested_by or context.user_id,
                context.run_id,
                context.thread_id,
                context.project_id,
                now,
                now,
            ),
        )
        self._event(approval_id, "created", redacted, requested_by or context.user_id)
        return self.get_approval_request(approval_id)

    def get_approval_request(self, approval_id: str) -> ApprovalRequest:
        row = self.sqlite.query_one("SELECT * FROM approval_requests WHERE approval_id = ?", (approval_id,))
        if row is None:
            raise BackendError("NOT_FOUND", "Approval request was not found.", {"approval_id": approval_id})
        return self._row_to_model(row)

    def list_pending_approvals(self, context: PolicyContext | None = None) -> list[ApprovalRequest]:
        rows = self.sqlite.query_all("SELECT * FROM approval_requests WHERE status = ? ORDER BY created_at", (ApprovalStatus.pending.value,))
        return [self._row_to_model(row) for row in rows]

    def resolve_approval_request(
        self,
        approval_id: str,
        decision: ApprovalDecision | str,
        edited_payload: JsonDict | None = None,
        decided_by: str | None = None,
        reason: str | None = None,
    ) -> ApprovalRequest:
        decision = ApprovalDecision(decision)
        status = {
            ApprovalDecision.approve: ApprovalStatus.approved,
            ApprovalDecision.edit: ApprovalStatus.edited,
            ApprovalDecision.reject: ApprovalStatus.rejected,
            ApprovalDecision.expire: ApprovalStatus.expired,
        }[decision]
        now = utc_now_iso()
        self.sqlite.execute(
            """
            UPDATE approval_requests
            SET status = ?, decided_by = ?, edited_payload_json = ?, reason = ?, updated_at = ?
            WHERE approval_id = ?
            """,
            (
                status.value,
                decided_by,
                dumps_json(self._redact(edited_payload)) if edited_payload is not None else None,
                reason,
                now,
                approval_id,
            ),
        )
        self._event(approval_id, decision.value, edited_payload or {}, decided_by)
        return self.get_approval_request(approval_id)

    def _event(self, approval_id: str, event_type: str, payload: JsonDict, actor: str | None) -> None:
        self.sqlite.execute(
            "INSERT INTO approval_events(approval_id, event_type, payload_json, actor, created_at) VALUES (?, ?, ?, ?, ?)",
            (approval_id, event_type, dumps_json(self._redact(payload)), actor, utc_now_iso()),
        )

    def _redact(self, value: JsonDict | None) -> JsonDict:
        if not value:
            return {}
        out: JsonDict = {}
        for key, item in value.items():
            if key.lower() in SENSITIVE_KEYS:
                out[key] = "[REDACTED]"
            elif isinstance(item, dict):
                out[key] = self._redact(item)
            else:
                out[key] = item
        return out

    def _row_to_model(self, row) -> ApprovalRequest:
        return ApprovalRequest(
            approval_id=row["approval_id"],
            action=row["action"],
            resource=row["resource"],
            payload=loads_json(row["payload_json"]),
            status=ApprovalStatus(row["status"]),
            requested_by=row["requested_by"],
            decided_by=row["decided_by"],
            edited_payload=loads_json(row["edited_payload_json"], None) if row["edited_payload_json"] else None,
            reason=row["reason"],
            run_id=row["run_id"],
            thread_id=row["thread_id"],
            project_id=row["project_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


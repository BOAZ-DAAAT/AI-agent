from __future__ import annotations

from data_agent_backend.models.approvals import ApprovalRequest, ApprovalStatus
from data_agent_backend.models.common import BackendError, JsonDict, utc_now_iso
from data_agent_backend.models.contexts import PolicyContext
from data_agent_backend.models.ids import UUID4IdGenerator
from data_agent_backend.models.runs import TERMINAL_RUN_STATUSES, RunEvent, RunRecord, RunStatus, RunSummary
from data_agent_backend.services.artifact_registry import ArtifactRegistry
from data_agent_backend.services.policy_engine import PolicyEngine
from data_agent_backend.storage.sqlite import SQLiteStore, dumps_json, loads_json


class RunService:
    def __init__(
        self,
        sqlite: SQLiteStore,
        policy_engine: PolicyEngine,
        artifact_registry: ArtifactRegistry,
        id_generator: UUID4IdGenerator | None = None,
    ) -> None:
        self.sqlite = sqlite
        self.policy_engine = policy_engine
        self.artifact_registry = artifact_registry
        self.id_generator = id_generator or UUID4IdGenerator()

    def create_run(
        self,
        run_id: str | None = None,
        thread_id: str | None = None,
        project_id: str | None = None,
        metadata: JsonDict | None = None,
        context: PolicyContext | None = None,
    ) -> RunRecord:
        context = context or PolicyContext(run_id=run_id, thread_id=thread_id, project_id=project_id)
        self.policy_engine.enforce("run.create", run_id or "runs", metadata or {}, context)
        run_id = run_id or self.id_generator.new_id("run")
        now = utc_now_iso()
        self.sqlite.execute(
            """
            INSERT INTO runs(run_id, thread_id, project_id, status, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, thread_id, project_id, RunStatus.created.value, dumps_json(metadata or {}), now, now),
        )
        return self.get_run(run_id, context)

    def get_run(self, run_id: str, context: PolicyContext | None = None) -> RunRecord:
        self.policy_engine.enforce("run.read", run_id, {}, context or PolicyContext(run_id=run_id))
        row = self.sqlite.query_one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        if row is None:
            raise BackendError("NOT_FOUND", "Run was not found.", {"run_id": run_id})
        return self._run_row_to_model(row)

    def list_runs(
        self,
        thread_id: str | None = None,
        project_id: str | None = None,
        status: RunStatus | str | None = None,
        context: PolicyContext | None = None,
    ) -> list[RunRecord]:
        self.policy_engine.enforce(
            "run.read",
            "runs",
            {"thread_id": thread_id, "project_id": project_id, "status": str(status) if status else None},
            context or PolicyContext(thread_id=thread_id, project_id=project_id),
        )
        normalized_status = self._normalize_status(status) if status is not None else None
        clauses: list[str] = []
        params: list[str] = []
        if thread_id:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if project_id:
            clauses.append("project_id = ?")
            params.append(project_id)
        if normalized_status:
            clauses.append("status = ?")
            params.append(normalized_status.value)

        sql = "SELECT * FROM runs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at"
        return [self._run_row_to_model(row) for row in self.sqlite.query_all(sql, params)]

    def update_status(
        self,
        run_id: str,
        status: RunStatus | str,
        metadata: JsonDict | None = None,
        context: PolicyContext | None = None,
    ) -> RunRecord:
        new_status = self._normalize_status(status)
        context = context or PolicyContext(run_id=run_id)
        self.policy_engine.enforce("run.update", run_id, {"status": new_status.value, "metadata": metadata or {}}, context)
        run = self.get_run(run_id, context)
        if run.status in TERMINAL_RUN_STATUSES:
            raise BackendError("RUN_TERMINAL", "Terminal runs cannot be updated.", {"run_id": run_id, "status": run.status.value})

        now = utc_now_iso()
        updates = ["status = ?", "updated_at = ?"]
        params: list[str] = [new_status.value, now]
        if metadata is not None:
            merged_metadata = dict(run.metadata)
            merged_metadata.update(metadata)
            updates.append("metadata_json = ?")
            params.append(dumps_json(merged_metadata))
        params.append(run_id)
        self.sqlite.execute(f"UPDATE runs SET {', '.join(updates)} WHERE run_id = ?", params)
        return self.get_run(run_id, context)

    def append_event(
        self,
        run_id: str,
        event_type: str,
        message: str,
        node_name: str | None = None,
        tool_name: str | None = None,
        artifact_ids: list[str] | None = None,
        approval_id: str | None = None,
        memory_ids: list[str] | None = None,
        metadata: JsonDict | None = None,
        context: PolicyContext | None = None,
    ) -> RunEvent:
        context = context or PolicyContext(run_id=run_id)
        self.policy_engine.enforce(
            "run.event.append",
            run_id,
            {"event_type": event_type, "node_name": node_name, "tool_name": tool_name},
            context,
        )
        self.get_run(run_id, context)
        event_id = self.id_generator.new_id("evt")
        now = utc_now_iso()
        self.sqlite.execute(
            """
            INSERT INTO run_events(
                event_id, run_id, event_type, message, node_name, tool_name, artifact_ids_json,
                approval_id, memory_ids_json, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                run_id,
                event_type,
                message,
                node_name,
                tool_name,
                dumps_json(artifact_ids or []),
                approval_id,
                dumps_json(memory_ids or []),
                dumps_json(metadata or {}),
                now,
            ),
        )
        return self._event_row_to_model(self.sqlite.query_one("SELECT * FROM run_events WHERE event_id = ?", (event_id,)))

    def list_events(self, run_id: str, context: PolicyContext | None = None) -> list[RunEvent]:
        context = context or PolicyContext(run_id=run_id)
        self.policy_engine.enforce("run.event.read", run_id, {}, context)
        self.get_run(run_id, context)
        rows = self.sqlite.query_all("SELECT * FROM run_events WHERE run_id = ? ORDER BY created_at", (run_id,))
        return [self._event_row_to_model(row) for row in rows]

    def get_summary(self, run_id: str, context: PolicyContext | None = None) -> RunSummary:
        context = context or PolicyContext(run_id=run_id)
        self.policy_engine.enforce("run.summary", run_id, {}, context)
        run = self.get_run(run_id, context)
        events = self.list_events(run_id, context)
        artifacts = self.artifact_registry.list_artifacts(run_id=run_id, context=context)
        approvals = self._list_pending_approvals_for_run(run_id)
        return RunSummary(run=run, events=events, artifacts=artifacts, pending_approvals=approvals)

    def _normalize_status(self, status: RunStatus | str) -> RunStatus:
        try:
            return status if isinstance(status, RunStatus) else RunStatus(status)
        except ValueError as exc:
            raise BackendError("VALIDATION_ERROR", "Invalid run status.", {"status": str(status)}) from exc

    def _run_row_to_model(self, row) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            thread_id=row["thread_id"],
            project_id=row["project_id"],
            status=RunStatus(row["status"]),
            metadata=loads_json(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _event_row_to_model(self, row) -> RunEvent:
        return RunEvent(
            event_id=row["event_id"],
            run_id=row["run_id"],
            event_type=row["event_type"],
            message=row["message"],
            node_name=row["node_name"],
            tool_name=row["tool_name"],
            artifact_ids=loads_json(row["artifact_ids_json"], []),
            approval_id=row["approval_id"],
            memory_ids=loads_json(row["memory_ids_json"], []),
            metadata=loads_json(row["metadata_json"]),
            created_at=row["created_at"],
        )

    def _list_pending_approvals_for_run(self, run_id: str) -> list[ApprovalRequest]:
        rows = self.sqlite.query_all(
            "SELECT * FROM approval_requests WHERE run_id = ? AND status = ? ORDER BY created_at",
            (run_id, ApprovalStatus.pending.value),
        )
        return [
            ApprovalRequest(
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
            for row in rows
        ]

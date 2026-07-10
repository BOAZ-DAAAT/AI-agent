from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from io import StringIO
from typing import Any

from data_agent_backend.config import BackendConfig
from data_agent_backend.models.approvals import ApprovalRequest
from data_agent_backend.models.artifacts import ArtifactRecord, ArtifactRef, ArtifactRegisterRequest, ArtifactType
from data_agent_backend.models.common import BackendError
from data_agent_backend.models.contexts import PolicyContext
from data_agent_backend.models.policy import PolicyDecision
from data_agent_backend.models.runs import RunEvent, RunRecord, RunStatus
from data_agent_backend.services.factory import BackendServices, create_backend_services


class BackendAdapter:
    """Thin orchestration-owned wrapper over the fixed data_agent_backend services."""

    def __init__(self, services: BackendServices | None = None, config: BackendConfig | None = None) -> None:
        self.services = services or create_backend_services(config)

    def create_run(
        self,
        *,
        thread_id: str | None = None,
        project_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunRecord:
        return self.services.run_service.create_run(thread_id=thread_id, project_id=project_id, metadata=metadata or {})

    def update_run_status(
        self,
        run_id: str,
        status: RunStatus | str,
        *,
        metadata: dict[str, Any] | None = None,
        context: PolicyContext | None = None,
    ) -> RunRecord:
        return self.services.run_service.update_status(run_id, status, metadata=metadata, context=context)

    def append_run_event(
        self,
        run_id: str,
        event_type: str,
        message: str,
        *,
        node_name: str | None = None,
        tool_name: str | None = None,
        artifact_ids: list[str] | None = None,
        approval_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        context: PolicyContext | None = None,
    ) -> RunEvent:
        return self.services.run_service.append_event(
            run_id,
            event_type,
            message,
            node_name=node_name,
            tool_name=tool_name,
            artifact_ids=artifact_ids,
            approval_id=approval_id,
            metadata=metadata,
            context=context,
        )

    def register_artifact(
        self,
        run_id: str,
        artifact_type: ArtifactType | str,
        *,
        content_text: str | None = None,
        content_bytes: bytes | None = None,
        filename: str,
        created_by_tool: str,
        context: PolicyContext | None = None,
        parent_ids: list[str] | None = None,
        lineage_edge_type: str = "derived_from",
        metadata: dict[str, Any] | None = None,
        preview: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        # 차트(PNG) 등 바이너리 아티팩트는 content_bytes 로 등록한다.
        # (하위 계층 registry/모델은 이미 지원 — 이 어댑터 통로만 열어준다.)
        if content_text is None and content_bytes is None:
            raise ValueError("register_artifact requires content_text or content_bytes.")
        artifact_type = ArtifactType(artifact_type)
        record = self.services.artifact_registry.register_artifact(
            ArtifactRegisterRequest(
                run_id=run_id,
                type=artifact_type,
                content_text=content_text,
                content_bytes=content_bytes,
                filename=filename,
                thread_id=context.thread_id if context else None,
                project_id=context.project_id if context else None,
                created_by_tool=created_by_tool,
                created_by_node=context.node_name if context else None,
                parent_ids=parent_ids or [],
                lineage_edge_type=lineage_edge_type,
                metadata=metadata or {},
                preview=preview,
                approval_id=context.approval_id if context else None,
            ),
            context or PolicyContext(run_id=run_id),
        )
        return record.ref()

    def get_artifact(self, artifact_id: str) -> ArtifactRecord:
        return self.services.artifact_registry.get_artifact(artifact_id)

    def list_artifacts(
        self,
        *,
        run_id: str,
        artifact_type: ArtifactType | str | None = None,
    ) -> list[ArtifactRecord]:
        return self.services.artifact_registry.list_artifacts(
            run_id=run_id,
            type=artifact_type,
        )

    def run_sql_preview(
        self,
        run_id: str,
        query: str,
        *,
        datasource_id: str | None = None,
        row_limit: int | None = None,
        context: PolicyContext | None = None,
    ) -> ArtifactRef:
        if datasource_id is not None:
            raise BackendError(
                "UNSUPPORTED_OPERATION",
                "Datasource-backed SQL execution no longer belongs to the backend core. Execute SQL in the agent runtime and register artifacts through the backend.",
                {"datasource_id": datasource_id},
            )
        return self._run_local_sql_preview(run_id, query, row_limit=row_limit, context=context)

    def _run_local_sql_preview(
        self,
        run_id: str,
        query: str,
        *,
        row_limit: int | None,
        context: PolicyContext | None,
    ) -> ArtifactRef:
        context = context or PolicyContext(run_id=run_id)
        row_limit = row_limit or self.services.config.default_sql_row_limit
        self._validate_local_preview_sql(query, row_limit)
        query_ref = self.register_artifact(
            run_id,
            "sql_query",
            content_text=query,
            filename="query.sql",
            created_by_tool="sql_agent.preview",
            context=context,
            metadata={"row_limit": row_limit},
        )
        rows, columns = self._execute_local_preview_query(query, row_limit)
        csv_text = self._rows_to_csv(columns, rows)
        return self.register_artifact(
            run_id,
            "sql_result",
            content_text=csv_text,
            filename="result.csv",
            created_by_tool="sql_agent.preview",
            context=context,
            parent_ids=[query_ref.artifact_id],
            lineage_edge_type="query_result_of",
            metadata={"row_limit": row_limit, "returned_rows": len(rows), "execution_owner": "agent_runtime_preview"},
            preview={"row_count": len(rows), "columns": columns, "sample_rows": [dict(zip(columns, r)) for r in rows[:5]]},
        )

    @staticmethod
    def _validate_local_preview_sql(query: str, row_limit: int) -> None:
        stripped = query.strip()
        if not stripped:
            raise BackendError("POLICY_BLOCKED", "SQL query is empty.")
        if row_limit <= 0:
            raise BackendError("POLICY_BLOCKED", "row_limit must be positive.")
        normalized = stripped.rstrip().rstrip(";")
        if ";" in normalized:
            raise BackendError("POLICY_BLOCKED", "Multiple-statement SQL is blocked.")
        first = stripped.lstrip().split(maxsplit=1)[0].upper()
        if first not in {"SELECT", "WITH"}:
            raise BackendError("POLICY_BLOCKED", "Only read-only SELECT queries are allowed.")
        blocked = {
            "INSERT", "UPDATE", "DELETE", "MERGE", "DROP", "ALTER",
            "CREATE", "TRUNCATE", "PRAGMA", "ATTACH", "DETACH",
            "INSTALL", "LOAD", "COPY", "EXPORT", "CALL",
        }
        tokens = {token.upper() for token in stripped.replace(";", " ").replace("\n", " ").split()}
        forbidden = sorted(tokens & blocked)
        if forbidden:
            raise BackendError("POLICY_BLOCKED", f"Blocked SQL keyword(s): {', '.join(forbidden)}.")

    @staticmethod
    def _execute_local_preview_query(query: str, row_limit: int) -> tuple[list[tuple[Any, ...]], list[str]]:
        try:
            with sqlite3.connect(":memory:") as conn:
                cursor = conn.execute(query)
                rows = cursor.fetchmany(row_limit)
                columns = [description[0] for description in (cursor.description or [])]
        except sqlite3.Error as exc:
            raise BackendError("POLICY_BLOCKED", f"Local SQL preview failed: {exc}") from exc
        return [tuple(row) for row in rows], columns

    @staticmethod
    def _rows_to_csv(columns: list[str], rows: list[tuple[Any, ...]]) -> str:
        if not columns:
            return ""
        buffer = StringIO()
        writer = csv.writer(buffer, lineterminator="\r\n")
        writer.writerow(columns)
        writer.writerows(rows)
        return buffer.getvalue()

    def check_policy(
        self,
        action: str,
        resource: str = "",
        payload: dict[str, Any] | None = None,
        context: PolicyContext | None = None,
    ) -> PolicyDecision:
        return self.services.policy_engine.evaluate(action, resource, payload or {}, context)

    def request_approval(
        self,
        action: str,
        resource: str,
        payload: dict[str, Any],
        *,
        context: PolicyContext | None = None,
        requested_by: str | None = None,
    ) -> ApprovalRequest:
        return self.services.approval_store.create_approval_request(action, resource, payload, context, requested_by)

    def get_approval_status(self, approval_id: str) -> ApprovalRequest:
        return self.services.approval_store.get_approval_request(approval_id)

    def save_workspace_file(
        self,
        run_id: str,
        markdown: str,
        *,
        context: PolicyContext | None = None,
        filename: str = "report.md",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        raise BackendError(
            "UNSUPPORTED_OPERATION",
            "Workspace-backed report persistence no longer belongs to the backend core. Register report artifacts directly.",
            {"run_id": run_id, "filename": filename},
        )

    def export_report(
        self,
        artifact_id: str,
        format: str = "md",
        *,
        destination: str | None = None,
        context: PolicyContext | None = None,
    ) -> ArtifactRef:
        raise BackendError(
            "UNSUPPORTED_OPERATION",
            "Export creation no longer belongs to the backend core. Handle report export outside the backend core.",
            {"artifact_id": artifact_id, "format": format, "destination": destination},
        )

    def register_ge_validation(
        self,
        run_id: str,
        *,
        table_name: str,
        source_ref: ArtifactRef,
        passed: bool,
        row_count: int,
        failed_expectations: list[dict[str, Any]] | None = None,
        schema_fingerprint: str = "unknown",
        context: PolicyContext | None = None,
    ) -> ArtifactRef:
        payload = {
            "run_id": run_id,
            "table_name": table_name,
            "source_ref": source_ref.model_dump(mode="json"),
            "generated_at": "adapter-generated",
            "suite_name": f"{table_name}_minimum_integrity",
            "expectation_summary": {
                "total": 3 + (1 if failed_expectations else 0),
                "failed": len(failed_expectations or []),
            },
            "passed": passed,
            "failed_expectations": failed_expectations or [],
            "row_count": row_count,
            "schema_fingerprint": schema_fingerprint,
            "upstream_artifact_refs": [source_ref.artifact_id],
        }
        return self.register_artifact(
            run_id,
            ArtifactType.file,
            content_text=json.dumps(payload, ensure_ascii=False, indent=2),
            filename=f"ge_{table_name}_{run_id}.json",
            created_by_tool="DATA_Analyst_Assistant_Agent.ge_validation",
            context=context,
            parent_ids=[source_ref.artifact_id],
            lineage_edge_type="validates",
            metadata={
                "kind": "ge_table_validation_json",
                "table_name": table_name,
                "schema_fingerprint": schema_fingerprint,
                "validates_artifact_id": source_ref.artifact_id,
            },
            preview={
                "table_name": table_name,
                "passed": passed,
                "failed_count": len(failed_expectations or []),
                "top_issues": failed_expectations[:3] if failed_expectations else [],
            },
        )

    def materialize_mart_metadata(
        self,
        run_id: str,
        *,
        mart_id: str,
        source_sql_artifact_id: str,
        approval_id: str,
        schema_json: dict[str, Any],
        refresh_policy: str,
        context: PolicyContext | None = None,
    ) -> ArtifactRef:
        payload = {
            "mart_id": mart_id,
            "owner": context.user_id if context else None,
            "run_id": run_id,
            "source_sql_artifact_id": source_sql_artifact_id,
            "schema_json": schema_json,
            "refresh_policy": refresh_policy,
            "lineage": [source_sql_artifact_id],
            "approval_id": approval_id,
            "created_at": "adapter-generated",
        }
        return self.register_artifact(
            run_id,
            ArtifactType.file,
            content_text=json.dumps(payload, ensure_ascii=False, indent=2),
            filename=f"mart_metadata_{mart_id}.json",
            created_by_tool="DATA_Analyst_Assistant_Agent.mart_metadata",
            context=context,
            parent_ids=[source_sql_artifact_id],
            lineage_edge_type="metadata_for",
            metadata={"kind": "mart_metadata", "mart_id": mart_id, "approval_id": approval_id},
            preview={"mart_id": mart_id, "refresh_policy": refresh_policy},
        )

    # ── Datasource orchestration methods ──

    def get_default_datasource_id(self) -> str | None:
        return None

    def get_catalog_summary(self, datasource_id: str) -> dict | None:
        return None

    def refresh_catalog(self, datasource_id: str) -> dict:
        raise BackendError(
            "UNSUPPORTED_OPERATION",
            "Catalog refresh is no longer provided by the backend core. Refresh schema/catalog directly from the agent runtime.",
            {"datasource_id": datasource_id},
        )

    def read_artifact_text(self, artifact_id: str) -> str:
        return self.services.artifact_store.read_text(artifact_id)

    def read_artifact_bytes(self, artifact_id: str) -> bytes:
        return self.services.artifact_store.read_bytes(artifact_id)

    @property
    def base_data_dir(self) -> Path:
        return self.services.config.base_data_dir

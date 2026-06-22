from __future__ import annotations

import shutil
from pathlib import Path

from data_agent_backend.models.artifacts import ArtifactRegisterRequest, ArtifactRef, ArtifactType
from data_agent_backend.models.common import BackendError, JsonDict, utc_now_iso
from data_agent_backend.models.contexts import PolicyContext
from data_agent_backend.models.ids import UUID4IdGenerator
from data_agent_backend.services.approval_store import ApprovalStore
from data_agent_backend.services.artifact_registry import ArtifactRegistry
from data_agent_backend.services.artifact_store import ArtifactStore
from data_agent_backend.services.policy_engine import PolicyEngine
from data_agent_backend.storage.filesystem import ensure_child_path, safe_filename
from data_agent_backend.storage.sqlite import SQLiteStore, dumps_json


class ExportService:
    def __init__(
        self,
        exports_dir: Path,
        sqlite: SQLiteStore,
        registry: ArtifactRegistry,
        store: ArtifactStore,
        policy_engine: PolicyEngine,
        approval_store: ApprovalStore,
        id_generator: UUID4IdGenerator | None = None,
    ) -> None:
        self.exports_dir = exports_dir
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite = sqlite
        self.registry = registry
        self.store = store
        self.policy_engine = policy_engine
        self.approval_store = approval_store
        self.id_generator = id_generator or UUID4IdGenerator()

    def create_export(
        self,
        artifact_id: str,
        format: str,
        destination: str | None = None,
        context: PolicyContext | None = None,
    ) -> ArtifactRef:
        context = context or PolicyContext()
        decision = self.policy_engine.evaluate(
            "export.create",
            destination or "/exports",
            {"artifact_id": artifact_id, "format": format, "destination": destination},
            context,
        )
        if decision.requires_approval:
            approval = self.approval_store.create_approval_request(
                "export.create",
                destination or "/exports",
                {"artifact_id": artifact_id, "format": format, "destination": destination},
                context,
            )
            raise BackendError("APPROVAL_REQUIRED", decision.reason, {"decision_id": decision.decision_id, "approval_id": approval.approval_id})
        if not decision.allowed:
            raise BackendError("POLICY_BLOCKED", decision.reason, {"decision_id": decision.decision_id})

        source = self.store.get_path(artifact_id)
        export_id = self.id_generator.new_id("exp")
        output_name = safe_filename(f"{export_id}.{format}")
        output_path = ensure_child_path(self.exports_dir, self.exports_dir / output_name)
        shutil.copyfile(source, output_path)
        export_artifact = self.registry.register_artifact(
            ArtifactRegisterRequest(
                run_id=context.run_id or "export",
                type=ArtifactType.export,
                content_bytes=output_path.read_bytes(),
                filename=output_name,
                created_by_tool=context.tool_name or "export_create",
                thread_id=context.thread_id,
                project_id=context.project_id,
                parent_ids=[artifact_id],
                lineage_edge_type="export_of",
                metadata={"format": format, "destination": destination, "export_path": str(output_path)},
                approval_id=context.approval_id,
            ),
            context,
        )
        self.sqlite.execute(
            """
            INSERT INTO exports(export_id, artifact_id, format, destination, output_path, approval_id, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (export_id, artifact_id, format, destination, str(output_path), context.approval_id, dumps_json({}), utc_now_iso()),
        )
        return export_artifact.ref()


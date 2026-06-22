from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable
from pathlib import Path

from data_agent_backend.models.artifacts import ArtifactRecord, ArtifactRegisterRequest, ArtifactType
from data_agent_backend.models.common import BackendError, JsonDict, utc_now_iso
from data_agent_backend.models.contexts import PolicyContext
from data_agent_backend.models.ids import UUID4IdGenerator
from data_agent_backend.services.artifact_store import ArtifactStore
from data_agent_backend.services.policy_engine import PolicyEngine
from data_agent_backend.storage.sqlite import SQLiteStore, dumps_json, loads_json


class ArtifactRegistry:
    def __init__(
        self,
        sqlite: SQLiteStore,
        artifact_store: ArtifactStore,
        policy_engine: PolicyEngine,
        id_generator: UUID4IdGenerator | None = None,
    ) -> None:
        self.sqlite = sqlite
        self.artifact_store = artifact_store
        self.policy_engine = policy_engine
        self.id_generator = id_generator or UUID4IdGenerator()

    def register_artifact(self, request: ArtifactRegisterRequest, context: PolicyContext | None = None) -> ArtifactRecord:
        context = context or PolicyContext(run_id=request.run_id)
        self.policy_engine.enforce(
            "artifact.register",
            f"artifact:{request.type.value}",
            {"type": request.type.value, "created_by_tool": request.created_by_tool},
            context,
        )
        if not request.created_by_tool or not request.run_id:
            raise BackendError("VALIDATION_ERROR", "run_id and created_by_tool are required for artifact registration.")
        if request.content_text is None and request.content_bytes is None:
            raise BackendError("VALIDATION_ERROR", "Artifact registration requires content.")

        artifact_id = self.id_generator.new_id("art")
        content = request.content_bytes if request.content_bytes is not None else request.content_text.encode("utf-8")
        uri, path, content_hash = self.artifact_store.put_bytes(artifact_id, content, request.filename or self._default_filename(request.type))
        if request.preview is not None:
            preview = request.preview
        else:
            try:
                preview = self._generate_preview(request.type, content, path)
            except Exception as exc:
                preview = {"preview_error": f"{type(exc).__name__}: {exc}"}
        metadata = dict(request.metadata)
        if "preview_error" in preview:
            metadata["preview_error"] = preview["preview_error"]
        now = utc_now_iso()
        with self.sqlite.connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts(
                    artifact_id, run_id, thread_id, project_id, type, uri, local_path, content_hash,
                    created_at, created_by_tool, created_by_node, parent_ids_json, metadata_json,
                    permissions_json, approval_id, policy_decision_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    request.run_id,
                    request.thread_id,
                    request.project_id,
                    request.type.value,
                    uri,
                    str(path),
                    content_hash,
                    now,
                    request.created_by_tool,
                    request.created_by_node,
                    dumps_json(request.parent_ids),
                    dumps_json(metadata),
                    dumps_json(request.permissions) if request.permissions is not None else None,
                    request.approval_id,
                    request.policy_decision_id,
                ),
            )
            conn.execute(
                "INSERT INTO artifact_previews(artifact_id, preview_json, created_at) VALUES (?, ?, ?)",
                (artifact_id, dumps_json(preview), now),
            )
            for parent_id in request.parent_ids:
                conn.execute(
                    """
                    INSERT INTO artifact_lineage(parent_id, child_id, edge_type, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (parent_id, artifact_id, request.lineage_edge_type, dumps_json({}), now),
                )
        return self.get_artifact(artifact_id)

    def get_artifact(self, artifact_id: str) -> ArtifactRecord:
        row = self.sqlite.query_one("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,))
        if row is None:
            raise BackendError("NOT_FOUND", "Artifact was not found.", {"artifact_id": artifact_id})
        preview = self.get_preview(artifact_id)
        return ArtifactRecord(
            artifact_id=row["artifact_id"],
            run_id=row["run_id"],
            thread_id=row["thread_id"],
            project_id=row["project_id"],
            type=ArtifactType(row["type"]),
            uri=row["uri"],
            local_path=row["local_path"],
            content_hash=row["content_hash"],
            created_at=row["created_at"],
            created_by_tool=row["created_by_tool"],
            created_by_node=row["created_by_node"],
            parent_ids=loads_json(row["parent_ids_json"], []),
            metadata=loads_json(row["metadata_json"]),
            preview=preview,
            permissions=loads_json(row["permissions_json"], None) if row["permissions_json"] else None,
            approval_id=row["approval_id"],
            policy_decision_id=row["policy_decision_id"],
        )

    def list_artifacts(
        self,
        run_id: str | None = None,
        thread_id: str | None = None,
        project_id: str | None = None,
        type: ArtifactType | str | None = None,
        context: PolicyContext | None = None,
    ) -> list[ArtifactRecord]:
        self.policy_engine.enforce("artifact.list", "artifacts", {}, context or PolicyContext())
        clauses: list[str] = []
        params: list[str] = []
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if thread_id:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if project_id:
            clauses.append("project_id = ?")
            params.append(project_id)
        if type:
            clauses.append("type = ?")
            params.append(str(type))
        sql = "SELECT artifact_id FROM artifacts"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at"
        return [self.get_artifact(row["artifact_id"]) for row in self.sqlite.query_all(sql, params)]

    def get_lineage(self, artifact_id: str) -> list[JsonDict]:
        rows = self.sqlite.query_all(
            "SELECT parent_id, child_id, edge_type, metadata_json, created_at FROM artifact_lineage WHERE child_id = ? OR parent_id = ?",
            (artifact_id, artifact_id),
        )
        return [
            {
                "parent_id": row["parent_id"],
                "child_id": row["child_id"],
                "edge_type": row["edge_type"],
                "metadata": loads_json(row["metadata_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_preview(self, artifact_id: str) -> JsonDict:
        row = self.sqlite.query_one("SELECT preview_json FROM artifact_previews WHERE artifact_id = ?", (artifact_id,))
        if row is None:
            return {}
        return loads_json(row["preview_json"])

    def attach_evidence(self, claim_id: str, artifact_ids: Iterable[str]) -> None:
        now = utc_now_iso()
        with self.sqlite.connect() as conn:
            for artifact_id in artifact_ids:
                conn.execute(
                    "INSERT OR REPLACE INTO claim_evidence(claim_id, artifact_id, metadata_json, created_at) VALUES (?, ?, ?, ?)",
                    (claim_id, artifact_id, dumps_json({}), now),
                )

    def validate_evidence(self, claims: dict[str, list[str]]) -> JsonDict:
        return {
            claim_id: all(self.sqlite.query_one("SELECT 1 FROM artifacts WHERE artifact_id = ?", (artifact_id,)) for artifact_id in ids)
            for claim_id, ids in claims.items()
        }

    def _default_filename(self, artifact_type: ArtifactType) -> str:
        suffix = "csv" if artifact_type in {ArtifactType.dataset, ArtifactType.sql_result} else "txt"
        return f"{artifact_type.value}.{suffix}"

    def _generate_preview(self, artifact_type: ArtifactType, content: bytes, path: Path) -> JsonDict:
        try:
            text = content[:200_000].decode("utf-8", errors="replace")
            if artifact_type in {ArtifactType.dataset, ArtifactType.sql_result}:
                reader = csv.DictReader(io.StringIO(text))
                rows = []
                null_counts: dict[str, int] = {}
                for idx, row in enumerate(reader):
                    if idx < 5:
                        rows.append(row)
                    for key, value in row.items():
                        if value in {"", None}:
                            null_counts[key] = null_counts.get(key, 0) + 1
                    if idx >= 99:
                        break
                return {
                    "row_count": len(rows) if text.count("\n") <= 6 else min(text.count("\n") - 1, 100),
                    "columns": reader.fieldnames or [],
                    "sample_rows": rows,
                    "null_counts": null_counts,
                }
            if artifact_type == ArtifactType.report:
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                return {"title": lines[0] if lines else "", "section_headings": [l for l in lines if l.startswith("#")][:10], "snippet": text[:500]}
            if artifact_type == ArtifactType.execution_log:
                return {"stdout_snippet": text[:1000], "stderr_snippet": "", "exit_code": None, "runtime_ms": None}
            if artifact_type == ArtifactType.chart:
                return {"chart_type": "unknown", "image_path": str(path)}
            return {"snippet": text[:1000]}
        except Exception as exc:
            return {"preview_error": f"{type(exc).__name__}: {exc}"}

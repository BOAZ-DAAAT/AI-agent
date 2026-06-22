from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_agent_backend.models.artifacts import ArtifactRecord, ArtifactRef, ArtifactRegisterRequest, ArtifactType
from data_agent_backend.models.common import BackendError, JsonDict
from data_agent_backend.models.contexts import PolicyContext
from data_agent_backend.services.artifact_registry import ArtifactRegistry
from data_agent_backend.services.artifact_store import ArtifactStore
from data_agent_backend.services.workspace_backend import WorkspaceBackend
from data_agent_backend.storage.filesystem import ensure_child_path


EXECUTION_LOG_FIELDS = {"status", "stdout", "stderr", "exit_code", "runtime_ms", "artifacts"}


class WorkspaceStorageService:
    def __init__(
        self,
        artifact_registry: ArtifactRegistry,
        artifact_store: ArtifactStore,
        workspace_backend: WorkspaceBackend,
    ) -> None:
        self.artifact_registry = artifact_registry
        self.artifact_store = artifact_store
        self.workspace_backend = workspace_backend

    def save_sql_query(
        self,
        run_id: str,
        query: str,
        *,
        context: PolicyContext | None = None,
        filename: str | None = None,
        metadata: JsonDict | None = None,
    ) -> ArtifactRef:
        return self._register_text(
            run_id,
            ArtifactType.sql_query,
            query,
            filename or "query.sql",
            "workspace_storage.save_sql_query",
            context,
            self._metadata(metadata, {"format": "sql"}),
        )

    def save_sql_result(
        self,
        run_id: str,
        rows: list[dict[str, Any]],
        *,
        context: PolicyContext | None = None,
        filename: str | None = None,
        metadata: JsonDict | None = None,
    ) -> ArtifactRef:
        columns = self._columns_from_rows(rows)
        preview = {"row_count": len(rows), "columns": columns, "sample_rows": rows[:5]}
        content = json.dumps(rows, ensure_ascii=False, sort_keys=True)
        return self._register_text(
            run_id,
            ArtifactType.sql_result,
            content,
            filename or "result.json",
            "workspace_storage.save_sql_result",
            context,
            self._metadata(metadata, {"format": "json_rows", "row_count": len(rows), "columns": columns}),
            preview=preview,
        )

    def save_python_code(
        self,
        run_id: str,
        code: str,
        *,
        context: PolicyContext | None = None,
        filename: str | None = None,
        metadata: JsonDict | None = None,
    ) -> ArtifactRef:
        return self._register_text(
            run_id,
            ArtifactType.python_code,
            code,
            filename or "code.py",
            "workspace_storage.save_python_code",
            context,
            self._metadata(metadata, {"format": "python"}),
        )

    def save_execution_log(
        self,
        run_id: str,
        log: JsonDict,
        *,
        context: PolicyContext | None = None,
        filename: str | None = None,
        metadata: JsonDict | None = None,
    ) -> ArtifactRef:
        filtered = {key: log[key] for key in EXECUTION_LOG_FIELDS if key in log}
        content = json.dumps(filtered, ensure_ascii=False, sort_keys=True)
        summary = {
            "format": "json",
            "status": filtered.get("status"),
            "exit_code": filtered.get("exit_code"),
            "runtime_ms": filtered.get("runtime_ms"),
        }
        preview = {
            "status": filtered.get("status"),
            "stdout_snippet": str(filtered.get("stdout", ""))[:1000],
            "stderr_snippet": str(filtered.get("stderr", ""))[:1000],
            "exit_code": filtered.get("exit_code"),
            "runtime_ms": filtered.get("runtime_ms"),
        }
        return self._register_text(
            run_id,
            ArtifactType.execution_log,
            content,
            filename or "execution_log.json",
            "workspace_storage.save_execution_log",
            context,
            self._metadata(metadata, summary),
            preview=preview,
        )

    def save_dataset(
        self,
        run_id: str,
        rows: list[dict[str, Any]] | str,
        *,
        context: PolicyContext | None = None,
        filename: str | None = None,
        metadata: JsonDict | None = None,
    ) -> ArtifactRef:
        if isinstance(rows, str):
            return self._register_text(
                run_id,
                ArtifactType.dataset,
                rows,
                filename or "dataset.csv",
                "workspace_storage.save_dataset",
                context,
                self._metadata(metadata, {"format": "csv"}),
            )
        columns = self._columns_from_rows(rows)
        preview = {"row_count": len(rows), "columns": columns, "sample_rows": rows[:5]}
        content = json.dumps(rows, ensure_ascii=False, sort_keys=True)
        return self._register_text(
            run_id,
            ArtifactType.dataset,
            content,
            filename or "dataset.json",
            "workspace_storage.save_dataset",
            context,
            self._metadata(metadata, {"format": "json_rows", "row_count": len(rows), "columns": columns}),
            preview=preview,
        )

    def save_chart(
        self,
        run_id: str,
        path: Path,
        *,
        allowed_base_dir: Path,
        context: PolicyContext | None = None,
        filename: str | None = None,
        metadata: JsonDict | None = None,
    ) -> ArtifactRef:
        source = ensure_child_path(allowed_base_dir, path)
        if not source.exists() or not source.is_file():
            raise BackendError("NOT_FOUND", "Chart file was not found.", {"path": str(path)})
        chart_filename = filename or f"chart{source.suffix or '.bin'}"
        preview = {"chart_type": source.suffix.removeprefix(".") or "unknown", "image_path": str(source)}
        return self._register_bytes(
            run_id,
            ArtifactType.chart,
            source.read_bytes(),
            chart_filename,
            "workspace_storage.save_chart",
            context,
            self._metadata(metadata, {"format": source.suffix.removeprefix(".") or "binary", "source_path": str(source)}),
            preview=preview,
        )

    def save_report(
        self,
        run_id: str,
        markdown: str,
        *,
        context: PolicyContext | None = None,
        filename: str | None = None,
        metadata: JsonDict | None = None,
    ) -> ArtifactRef:
        return self._register_text(
            run_id,
            ArtifactType.report,
            markdown,
            filename or "report.md",
            "workspace_storage.save_report",
            context,
            self._metadata(metadata, {"format": "markdown"}),
        )

    def get_artifact(self, artifact_id: str) -> ArtifactRecord:
        return self.artifact_registry.get_artifact(artifact_id)

    def list_run_artifacts(self, run_id: str, *, context: PolicyContext | None = None) -> list[ArtifactRecord]:
        return self.artifact_registry.list_artifacts(run_id=run_id, context=context)

    def preview_artifact(self, artifact_id: str) -> JsonDict:
        return self.artifact_registry.get_preview(artifact_id)

    def _register_text(
        self,
        run_id: str,
        artifact_type: ArtifactType,
        content: str,
        filename: str,
        created_by_tool: str,
        context: PolicyContext | None,
        metadata: JsonDict,
        *,
        preview: JsonDict | None = None,
    ) -> ArtifactRef:
        return self.artifact_registry.register_artifact(
            ArtifactRegisterRequest(
                run_id=run_id,
                type=artifact_type,
                content_text=content,
                filename=filename,
                created_by_tool=created_by_tool,
                thread_id=context.thread_id if context else None,
                project_id=context.project_id if context else None,
                metadata=metadata,
                preview=preview,
                approval_id=context.approval_id if context else None,
            ),
            context or PolicyContext(run_id=run_id),
        ).ref()

    def _register_bytes(
        self,
        run_id: str,
        artifact_type: ArtifactType,
        content: bytes,
        filename: str,
        created_by_tool: str,
        context: PolicyContext | None,
        metadata: JsonDict,
        *,
        preview: JsonDict | None = None,
    ) -> ArtifactRef:
        return self.artifact_registry.register_artifact(
            ArtifactRegisterRequest(
                run_id=run_id,
                type=artifact_type,
                content_bytes=content,
                filename=filename,
                created_by_tool=created_by_tool,
                thread_id=context.thread_id if context else None,
                project_id=context.project_id if context else None,
                metadata=metadata,
                preview=preview,
                approval_id=context.approval_id if context else None,
            ),
            context or PolicyContext(run_id=run_id),
        ).ref()

    def _metadata(self, provided: JsonDict | None, standard: JsonDict) -> JsonDict:
        merged = dict(provided or {})
        merged.update(standard)
        return merged

    def _columns_from_rows(self, rows: list[dict[str, Any]]) -> list[str]:
        columns: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for column in row:
                if column not in seen:
                    seen.add(column)
                    columns.append(column)
        return columns

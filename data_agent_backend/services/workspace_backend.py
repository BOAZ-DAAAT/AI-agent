from __future__ import annotations

from data_agent_backend.models.common import BackendError
from data_agent_backend.models.contexts import PolicyContext
from data_agent_backend.models.workspace import WorkspaceEntry, WorkspacePreview, WorkspaceWriteResult
from data_agent_backend.services.workspace_router import WorkspaceRouter, normalize_logical_path


class WorkspaceBackend:
    def __init__(self, router: WorkspaceRouter) -> None:
        self.router = router

    def list(self, path: str = "/", context: PolicyContext | None = None) -> list[WorkspaceEntry]:
        context = context or PolicyContext()
        normalized = normalize_logical_path(path)
        if normalized == "/":
            return self.router.root_entries()
        _, mount, _ = self.router.resolve(normalized)
        return mount.list(normalized, context)

    def read_text(self, path: str, context: PolicyContext | None = None) -> str:
        context = context or PolicyContext()
        normalized, mount, _ = self.router.resolve(path)
        return mount.read_text(normalized, context)

    def write_text(self, path: str, content: str, context: PolicyContext | None = None) -> WorkspaceWriteResult:
        context = context or PolicyContext()
        normalized, mount, _ = self.router.resolve(path)
        return mount.write_text(normalized, content, context)

    def exists(self, path: str, context: PolicyContext | None = None) -> bool:
        try:
            normalized, mount, _ = self.router.resolve(path)
            return mount.exists(normalized, context or PolicyContext())
        except BackendError:
            return False

    def preview(self, path_or_artifact_ref: str, context: PolicyContext | None = None) -> WorkspacePreview:
        context = context or PolicyContext()
        if not path_or_artifact_ref.startswith("/"):
            path_or_artifact_ref = f"/artifacts/{path_or_artifact_ref}"
        normalized, mount, _ = self.router.resolve(path_or_artifact_ref)
        return mount.preview(normalized, context)


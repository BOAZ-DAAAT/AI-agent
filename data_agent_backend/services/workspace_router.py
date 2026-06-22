from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Protocol

from data_agent_backend.models.common import BackendError
from data_agent_backend.models.contexts import PolicyContext
from data_agent_backend.models.workspace import WorkspaceEntry, WorkspaceEntryType, WorkspacePreview, WorkspaceWriteResult
from data_agent_backend.services.artifact_registry import ArtifactRegistry
from data_agent_backend.services.artifact_store import ArtifactStore
from data_agent_backend.services.catalog_store import CatalogStore
from data_agent_backend.services.memory_store import MemoryStore
from data_agent_backend.services.policy_engine import PolicyEngine
from data_agent_backend.storage.filesystem import ensure_child_path


class MountBackend(Protocol):
    def list(self, path: str, context: PolicyContext) -> list[WorkspaceEntry]: ...
    def read_text(self, path: str, context: PolicyContext) -> str: ...
    def write_text(self, path: str, content: str, context: PolicyContext) -> WorkspaceWriteResult: ...
    def exists(self, path: str, context: PolicyContext) -> bool: ...
    def preview(self, path: str, context: PolicyContext) -> WorkspacePreview: ...


def normalize_logical_path(path: str) -> str:
    if "\x00" in path:
        raise BackendError("POLICY_BLOCKED", "Null bytes are not allowed in logical paths.", {"path": path})
    path = path.replace("\\", "/")
    if ":" in path.split("/", 1)[0]:
        raise BackendError("POLICY_BLOCKED", "Host absolute paths are not allowed.", {"path": path})
    if not path.startswith("/"):
        path = "/" + path
    pure = PurePosixPath(path)
    parts = [part for part in pure.parts if part != "/"]
    lowered = [part.lower() for part in parts]
    if any(part in {"..", "~"} for part in parts):
        raise BackendError("POLICY_BLOCKED", "Path traversal is not allowed.", {"path": path})
    if lowered and lowered[0] == "secrets":
        raise BackendError("POLICY_BLOCKED", "Secret access is blocked.", {"path": path})
    if "secrets" in lowered:
        raise BackendError("POLICY_BLOCKED", "Secret traversal attempts are blocked.", {"path": path})
    return "/" + "/".join(parts) if parts else "/"


class WorkspaceRouter:
    def __init__(self, mounts: dict[str, MountBackend]) -> None:
        self.mounts = mounts

    def resolve(self, path: str) -> tuple[str, MountBackend, str]:
        normalized = normalize_logical_path(path)
        if normalized == "/":
            raise BackendError("VALIDATION_ERROR", "Root path resolves to mount listing, not a concrete mount.")
        mount_name = "/" + normalized.strip("/").split("/", 1)[0]
        mount = self.mounts.get(mount_name)
        if mount is None:
            raise BackendError("NOT_FOUND", "Logical mount was not found.", {"path": normalized})
        return normalized, mount, mount_name

    def root_entries(self) -> list[WorkspaceEntry]:
        return [
            WorkspaceEntry(path=name, name=name.removeprefix("/"), type=WorkspaceEntryType.mount)
            for name in sorted(self.mounts)
        ]


class LocalDirectoryMount:
    def __init__(self, mount_name: str, base_dir: Path, policy_engine: PolicyEngine, writable: bool) -> None:
        self.mount_name = mount_name
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.policy_engine = policy_engine
        self.writable = writable

    def _host_path(self, path: str) -> Path:
        rel = normalize_logical_path(path).removeprefix(self.mount_name).strip("/")
        target = ensure_child_path(self.base_dir, self.base_dir / rel)
        if target.exists():
            ensure_child_path(self.base_dir, target.resolve())
        return target

    def list(self, path: str, context: PolicyContext) -> list[WorkspaceEntry]:
        self.policy_engine.enforce("workspace.list", path, {}, context)
        host = self._host_path(path)
        if not host.exists():
            raise BackendError("NOT_FOUND", "Workspace path was not found.", {"path": path})
        if host.is_file():
            host = host.parent
        entries = []
        for item in sorted(host.iterdir(), key=lambda p: p.name):
            ensure_child_path(self.base_dir, item.resolve())
            entries.append(
                WorkspaceEntry(
                    path=f"{self.mount_name}/{item.relative_to(self.base_dir).as_posix()}",
                    name=item.name,
                    type=WorkspaceEntryType.directory if item.is_dir() else WorkspaceEntryType.file,
                    size=item.stat().st_size if item.is_file() else None,
                )
            )
        return entries

    def read_text(self, path: str, context: PolicyContext) -> str:
        self.policy_engine.enforce("workspace.read", path, {}, context)
        host = self._host_path(path)
        if not host.exists() or not host.is_file():
            raise BackendError("NOT_FOUND", "Workspace file was not found.", {"path": path})
        return host.read_text(encoding="utf-8")

    def write_text(self, path: str, content: str, context: PolicyContext) -> WorkspaceWriteResult:
        self.policy_engine.enforce("workspace.write", path, {"bytes": len(content.encode("utf-8"))}, context)
        if not self.writable:
            raise BackendError("POLICY_BLOCKED", "This mount is read-only.", {"path": path})
        host = self._host_path(path)
        existed = host.exists()
        host.parent.mkdir(parents=True, exist_ok=True)
        ensure_child_path(self.base_dir, host.parent.resolve())
        host.write_text(content, encoding="utf-8")
        return WorkspaceWriteResult(path=normalize_logical_path(path), bytes_written=len(content.encode("utf-8")), overwritten=existed)

    def exists(self, path: str, context: PolicyContext) -> bool:
        try:
            return self._host_path(path).exists()
        except BackendError:
            return False

    def preview(self, path: str, context: PolicyContext) -> WorkspacePreview:
        text = self.read_text(path, context)
        return WorkspacePreview(path=normalize_logical_path(path), kind="text", preview={"snippet": text[:1000], "bytes": len(text.encode("utf-8"))})


class ReadOnlyPolicyMount(LocalDirectoryMount):
    def write_text(self, path: str, content: str, context: PolicyContext) -> WorkspaceWriteResult:
        self.policy_engine.enforce("workspace.write", path, {"bytes": len(content.encode("utf-8"))}, context)
        raise BackendError("POLICY_BLOCKED", "This mount is read-only.", {"path": path})


class ArtifactMount:
    def __init__(self, registry: ArtifactRegistry, store: ArtifactStore, policy_engine: PolicyEngine) -> None:
        self.registry = registry
        self.store = store
        self.policy_engine = policy_engine

    def _artifact_id(self, path: str) -> str | None:
        rest = normalize_logical_path(path).removeprefix("/artifacts").strip("/")
        return rest.split("/", 1)[0] if rest else None

    def list(self, path: str, context: PolicyContext) -> list[WorkspaceEntry]:
        self.policy_engine.enforce("workspace.list", path, {}, context)
        return [
            WorkspaceEntry(path=f"/artifacts/{artifact.artifact_id}", name=artifact.artifact_id, type=WorkspaceEntryType.artifact, metadata={"type": artifact.type.value})
            for artifact in self.registry.list_artifacts(context=context)
        ]

    def read_text(self, path: str, context: PolicyContext) -> str:
        self.policy_engine.enforce("workspace.read", path, {}, context)
        artifact_id = self._artifact_id(path)
        if not artifact_id:
            raise BackendError("VALIDATION_ERROR", "Artifact ID is required.", {"path": path})
        return self.store.read_text(artifact_id)

    def write_text(self, path: str, content: str, context: PolicyContext) -> WorkspaceWriteResult:
        self.policy_engine.enforce("workspace.write", path, {"bytes": len(content.encode("utf-8"))}, context)
        raise BackendError("POLICY_BLOCKED", "Direct write to /artifacts is not allowed.", {"path": path})

    def exists(self, path: str, context: PolicyContext) -> bool:
        artifact_id = self._artifact_id(path)
        return bool(artifact_id and self.store.exists(artifact_id))

    def preview(self, path: str, context: PolicyContext) -> WorkspacePreview:
        self.policy_engine.enforce("artifact.preview", path, {}, context)
        artifact_id = self._artifact_id(path) or path
        return WorkspacePreview(path=f"/artifacts/{artifact_id}", kind="artifact", preview=self.registry.get_preview(artifact_id))


class MemoryMount:
    def __init__(self, memory_store: MemoryStore, policy_engine: PolicyEngine) -> None:
        self.memory_store = memory_store
        self.policy_engine = policy_engine

    def list(self, path: str, context: PolicyContext) -> list[WorkspaceEntry]:
        self.policy_engine.enforce("workspace.list", path, {}, context)
        namespace = self._namespace(context)
        return [
            WorkspaceEntry(path=f"/memory/{record.memory_id}", name=record.memory_id, type=WorkspaceEntryType.memory, metadata={"type": record.type.value})
            for record in self.memory_store.list_memory(namespace)
        ]

    def read_text(self, path: str, context: PolicyContext) -> str:
        self.policy_engine.enforce("workspace.read", path, {}, context)
        memory_id = normalize_logical_path(path).removeprefix("/memory").strip("/")
        record = self.memory_store.get_memory(memory_id)
        if record.status.value != "active":
            raise BackendError("NOT_FOUND", "Only active memory is exposed through /memory.", {"memory_id": memory_id})
        return record.content_text

    def write_text(self, path: str, content: str, context: PolicyContext) -> WorkspaceWriteResult:
        self.policy_engine.enforce("workspace.write", path, {"bytes": len(content.encode("utf-8"))}, context)
        raise BackendError("APPROVAL_REQUIRED", "Direct memory writes require approval.", {"path": path})

    def exists(self, path: str, context: PolicyContext) -> bool:
        try:
            self.read_text(path, context)
            return True
        except BackendError:
            return False

    def preview(self, path: str, context: PolicyContext) -> WorkspacePreview:
        text = self.read_text(path, context)
        return WorkspacePreview(path=normalize_logical_path(path), kind="memory", preview={"snippet": text[:1000]})

    def _namespace(self, context: PolicyContext) -> list[str]:
        namespace = []
        if context.user_id:
            namespace += ["user", context.user_id]
        if context.project_id:
            namespace += ["project", context.project_id]
        return namespace or ["global"]


from __future__ import annotations

from pathlib import Path

from data_agent_backend.models.common import BackendError
from data_agent_backend.models.contexts import PolicyContext
from data_agent_backend.models.workspace import WorkspaceEntry, WorkspaceEntryType
from data_agent_backend.services.policy_engine import PolicyEngine
from data_agent_backend.storage.filesystem import ensure_child_path


class CatalogStore:
    def __init__(self, base_dir: Path, policy_engine: PolicyEngine) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.policy_engine = policy_engine

    def list(self, context: PolicyContext | None = None) -> list[WorkspaceEntry]:
        self.policy_engine.enforce("catalog.read", "/catalog", {}, context or PolicyContext())
        return [
            WorkspaceEntry(
                path=f"/catalog/{item.name}",
                name=item.name,
                type=WorkspaceEntryType.directory if item.is_dir() else WorkspaceEntryType.file,
                size=item.stat().st_size if item.is_file() else None,
            )
            for item in sorted(self.base_dir.iterdir(), key=lambda p: p.name)
        ]

    def read_text(self, path_or_name: str, context: PolicyContext | None = None) -> str:
        self.policy_engine.enforce("catalog.read", f"/catalog/{path_or_name}", {}, context or PolicyContext())
        rel = path_or_name.removeprefix("/catalog/").strip("/")
        path = ensure_child_path(self.base_dir, self.base_dir / rel)
        if not path.exists() or not path.is_file():
            raise BackendError("NOT_FOUND", "Catalog entry was not found.", {"path": path_or_name})
        return path.read_text(encoding="utf-8")


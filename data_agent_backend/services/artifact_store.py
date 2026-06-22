from __future__ import annotations

import hashlib
from pathlib import Path

from data_agent_backend.models.common import BackendError
from data_agent_backend.storage.filesystem import ensure_child_path, safe_filename


class ArtifactStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def put_text(self, artifact_id: str, content: str, filename: str | None = None) -> tuple[str, Path, str]:
        return self.put_bytes(artifact_id, content.encode("utf-8"), filename or "content.txt")

    def put_bytes(self, artifact_id: str, content: bytes, filename: str | None = None) -> tuple[str, Path, str]:
        artifact_dir = ensure_child_path(self.base_dir, self.base_dir / artifact_id)
        artifact_dir.mkdir(parents=True, exist_ok=False)
        path = ensure_child_path(artifact_dir, artifact_dir / safe_filename(filename))
        if path.exists():
            raise BackendError("ALREADY_EXISTS", "Artifact content is append-only and cannot be overwritten.", {"artifact_id": artifact_id})
        path.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        return path.resolve().as_uri(), path, digest

    def get_path(self, artifact_id: str) -> Path:
        artifact_dir = ensure_child_path(self.base_dir, self.base_dir / artifact_id)
        if not artifact_dir.exists():
            raise BackendError("NOT_FOUND", "Artifact content was not found.", {"artifact_id": artifact_id})
        files = [item for item in artifact_dir.iterdir() if item.is_file()]
        if not files:
            raise BackendError("NOT_FOUND", "Artifact content was not found.", {"artifact_id": artifact_id})
        return ensure_child_path(artifact_dir, files[0])

    def read_text(self, artifact_id: str) -> str:
        return self.get_path(artifact_id).read_text(encoding="utf-8")

    def read_bytes(self, artifact_id: str) -> bytes:
        return self.get_path(artifact_id).read_bytes()

    def exists(self, artifact_id: str) -> bool:
        try:
            self.get_path(artifact_id)
            return True
        except BackendError:
            return False


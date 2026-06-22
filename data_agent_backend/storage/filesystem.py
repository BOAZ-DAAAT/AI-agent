from __future__ import annotations

import re
from pathlib import Path

from data_agent_backend.models.common import BackendError


def safe_filename(name: str | None, default: str = "content.bin") -> str:
    raw = name or default
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return cleaned or default


def ensure_child_path(base: Path, path: Path) -> Path:
    base_resolved = base.resolve()
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise BackendError("POLICY_BLOCKED", "Path escapes its configured storage root.", {"path": str(path)}) from exc
    return resolved


from __future__ import annotations

from data_agent_backend.models.common import JsonDict, utc_now_iso
from data_agent_backend.storage.sqlite import SQLiteStore, dumps_json


class CheckpointManager:
    def __init__(self, sqlite: SQLiteStore) -> None:
        self.sqlite = sqlite

    def ensure_run(self, run_id: str, thread_id: str | None = None, project_id: str | None = None, metadata: JsonDict | None = None) -> None:
        now = utc_now_iso()
        self.sqlite.execute(
            """
            INSERT INTO runs(run_id, thread_id, project_id, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (run_id, thread_id, project_id, dumps_json(metadata or {}), now, now),
        )


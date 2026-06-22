from __future__ import annotations

import json
import re
from typing import Any

from data_agent_backend.models.common import BackendError, JsonDict, utc_now_iso
from data_agent_backend.models.contexts import PolicyContext
from data_agent_backend.models.ids import UUID4IdGenerator
from data_agent_backend.models.memory import MemoryRecord, MemoryStatus, MemoryType
from data_agent_backend.services.approval_store import ApprovalStore
from data_agent_backend.services.policy_engine import PolicyEngine
from data_agent_backend.storage.sqlite import SQLiteStore, dumps_json, loads_json


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d[\d .-]{7,}\d)\b")
PII_KEYS = {"email", "phone", "ssn", "address", "name", "first_name", "last_name", "birthdate"}


class MemoryStore:
    def __init__(
        self,
        sqlite: SQLiteStore,
        policy_engine: PolicyEngine,
        approval_store: ApprovalStore,
        id_generator: UUID4IdGenerator | None = None,
    ) -> None:
        self.sqlite = sqlite
        self.policy_engine = policy_engine
        self.approval_store = approval_store
        self.id_generator = id_generator or UUID4IdGenerator()

    def propose_memory(
        self,
        namespace: list[str],
        type: MemoryType | str,
        content: str | JsonDict | list[Any],
        source: JsonDict,
        metadata: JsonDict | None = None,
        context: PolicyContext | None = None,
    ) -> MemoryRecord:
        context = context or PolicyContext()
        self.policy_engine.enforce("memory.propose", "/memory", {"namespace": namespace, "type": str(type)}, context)
        memory_type = MemoryType(type)
        content_text, content_json = self._split_content(content)
        flags = self._risk_flags(content_text, content_json)
        approval = self.approval_store.create_approval_request(
            "memory.write",
            "/memory",
            {"namespace": namespace, "type": memory_type.value, "content_text": content_text, "metadata": metadata or {}, "risk_flags": flags},
            context,
        )
        now = utc_now_iso()
        memory_id = self.id_generator.new_id("mem")
        self.sqlite.execute(
            """
            INSERT INTO memory_records(
                memory_id, namespace_json, type, status, content_text, content_json,
                source_json, metadata_json, approval_id, risk_flags_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                dumps_json(namespace),
                memory_type.value,
                MemoryStatus.pending.value,
                content_text,
                dumps_json(content_json) if content_json is not None else None,
                dumps_json(source),
                dumps_json(metadata or {}),
                approval.approval_id,
                dumps_json(flags),
                now,
                now,
            ),
        )
        return self.get_memory(memory_id)

    def list_memory(self, namespace: list[str], type: MemoryType | str | None = None, active_only: bool = True) -> list[MemoryRecord]:
        clauses = ["namespace_json = ?"]
        params: list[Any] = [dumps_json(namespace)]
        if type:
            clauses.append("type = ?")
            params.append(MemoryType(type).value)
        if active_only:
            clauses.append("status = ?")
            params.append(MemoryStatus.active.value)
        rows = self.sqlite.query_all(f"SELECT * FROM memory_records WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC", params)
        return [self._row_to_model(row) for row in rows]

    def get_memory(self, memory_id: str) -> MemoryRecord:
        row = self.sqlite.query_one("SELECT * FROM memory_records WHERE memory_id = ?", (memory_id,))
        if row is None:
            raise BackendError("NOT_FOUND", "Memory record was not found.", {"memory_id": memory_id})
        return self._row_to_model(row)

    def approve_memory(self, memory_id: str, approved_by: str | None = None) -> MemoryRecord:
        record = self.get_memory(memory_id)
        self.approval_store.resolve_approval_request(record.approval_id or "", "approve", decided_by=approved_by)
        return self._set_status(memory_id, MemoryStatus.active)

    def reject_memory(self, memory_id: str, rejected_by: str | None = None, reason: str | None = None) -> MemoryRecord:
        record = self.get_memory(memory_id)
        self.approval_store.resolve_approval_request(record.approval_id or "", "reject", decided_by=rejected_by, reason=reason)
        return self._set_status(memory_id, MemoryStatus.rejected)

    def archive_memory(self, memory_id: str, archived_by: str | None = None, reason: str | None = None) -> MemoryRecord:
        return self._set_status(memory_id, MemoryStatus.archived)

    def search_memory(self, query: str, namespace: list[str], type: MemoryType | str | None = None) -> list[MemoryRecord]:
        clauses = ["namespace_json = ?", "status = ?", "content_text LIKE ?"]
        params: list[Any] = [dumps_json(namespace), MemoryStatus.active.value, f"%{query}%"]
        if type:
            clauses.append("type = ?")
            params.append(MemoryType(type).value)
        rows = self.sqlite.query_all(f"SELECT * FROM memory_records WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC", params)
        return [self._row_to_model(row) for row in rows]

    def _set_status(self, memory_id: str, status: MemoryStatus) -> MemoryRecord:
        self.sqlite.execute("UPDATE memory_records SET status = ?, updated_at = ? WHERE memory_id = ?", (status.value, utc_now_iso(), memory_id))
        return self.get_memory(memory_id)

    def _split_content(self, content: str | JsonDict | list[Any]) -> tuple[str, Any | None]:
        if isinstance(content, str):
            return content, None
        return json.dumps(content, ensure_ascii=False, sort_keys=True), content

    def _risk_flags(self, content_text: str, content_json: Any | None) -> list[str]:
        flags: list[str] = []
        if len(content_text) > 4000:
            flags.append("too_long")
        if EMAIL_RE.search(content_text):
            flags.append("email_pattern")
        if PHONE_RE.search(content_text):
            flags.append("phone_pattern")
        if content_text.count("\n") >= 5 and "," in content_text:
            flags.append("csv_like_rows")
        if content_json is not None:
            if isinstance(content_json, list):
                flags.append("json_array_records")
            keys = set()
            if isinstance(content_json, dict):
                keys = {str(k).lower() for k in content_json.keys()}
            if keys & PII_KEYS:
                flags.append("pii_like_field_names")
            if isinstance(content_json, dict) and any(isinstance(value, list) for value in content_json.values()):
                flags.append("dataframe_like_structure")
        return sorted(set(flags))

    def _row_to_model(self, row) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row["memory_id"],
            namespace=loads_json(row["namespace_json"], []),
            type=MemoryType(row["type"]),
            status=MemoryStatus(row["status"]),
            content_text=row["content_text"],
            content_json=loads_json(row["content_json"], None) if row["content_json"] else None,
            source=loads_json(row["source_json"]),
            metadata=loads_json(row["metadata_json"]),
            approval_id=row["approval_id"],
            risk_flags=loads_json(row["risk_flags_json"], []),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

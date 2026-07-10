from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from data_agent_backend.models.common import BackendError, JsonDict, utc_now_iso
from data_agent_backend.models.integrity import IntegrityCheckResult, IntegrityJobStatus, IntegrityStatus
from data_agent_backend.models.tool_results import ToolResult
from data_agent_backend.storage.sqlite import SQLiteStore, dumps_json, loads_json


READY_STATUSES = {IntegrityStatus.pass_.value, IntegrityStatus.warning.value}
SUCCESS_STATUSES = {IntegrityStatus.pass_.value, IntegrityStatus.warning.value}
ATTENTION_STATUSES = {IntegrityStatus.fail.value, IntegrityStatus.warning.value, IntegrityStatus.stale.value, IntegrityStatus.pending.value, IntegrityStatus.running.value}


class IntegrityRunner(Protocol):
    def run_checks(
        self,
        dataset_name: str,
        tables: list[str] | None,
        source_version: str | None,
        metadata: JsonDict | None = None,
    ) -> list[IntegrityCheckResult]: ...


class DefaultIntegrityRunner:
    """Safe fallback runner.

    It does not assume access to the ingested database engine. When callers pass
    row-count metadata (as storage ingest does), it verifies the declared tables
    were copied with non-negative counts. Otherwise it records a warning so the
    dataset is not silently treated as fully validated.
    """

    def run_checks(
        self,
        dataset_name: str,
        tables: list[str] | None,
        source_version: str | None,
        metadata: JsonDict | None = None,
    ) -> list[IntegrityCheckResult]:
        metadata = metadata or {}
        try:
            from DATA_Analyst_Assistant_Agent.agents.sql.db.integrity_runner import run_dataset_integrity_checks
        except Exception:
            run_dataset_integrity_checks = None

        if run_dataset_integrity_checks is not None:
            try:
                payload = run_dataset_integrity_checks(
                    dataset_name=dataset_name,
                    table_scope=tables,
                    source_version=source_version,
                    metadata=metadata or {},
                )
                results: list[IntegrityCheckResult] = []
                for table_name, summary in payload.get("tables", {}).items():
                    status = str(summary.get("status", "warning")).lower()
                    enum_status = IntegrityStatus(status if status != "pass_" else "pass")
                    severity = "error" if enum_status == IntegrityStatus.fail else "warning" if enum_status == IntegrityStatus.warning else "info"
                    artifact_refs = payload.get("artifact_refs", [])
                    results.append(
                        IntegrityCheckResult(
                            table_name=table_name,
                            status=enum_status,
                            severity=severity,
                            summary={
                                "physical_status": summary.get("physical_status"),
                                "semantic_status": summary.get("semantic_status"),
                                "failures": summary.get("failures", []),
                                "temporary_fix_artifacts": summary.get("temporary_fix_artifacts", []),
                                "root_cause_summary": summary.get("root_cause_summary", []),
                            },
                            artifact_refs=artifact_refs,
                            metadata={
                                "runner": "great_expectations",
                                "source_version": source_version,
                            },
                        )
                    )
                if results:
                    return results
            except Exception as exc:
                raise BackendError(
                    "INTEGRITY_RUN_FAILED",
                    "Great Expectations runner execution failed.",
                    {
                        "dataset_name": dataset_name,
                        "tables": list(tables or []),
                        "source_version": source_version,
                        "runner_error": f"{type(exc).__name__}: {exc}",
                    },
                ) from exc

        row_counts = metadata.get("row_counts") if isinstance(metadata.get("row_counts"), dict) else {}
        table_names = list(tables or row_counts.keys())
        if not table_names:
            return [
                IntegrityCheckResult(
                    table_name=None,
                    status=IntegrityStatus.warning,
                    severity="warning",
                    summary={"message": "No table list or executable integrity runner was provided."},
                    metadata={"runner": "default", "dataset_name": dataset_name, "source_version": source_version},
                )
            ]

        results: list[IntegrityCheckResult] = []
        for table in table_names:
            row_count = row_counts.get(table)
            if row_count is None:
                results.append(
                    IntegrityCheckResult(
                        table_name=table,
                        status=IntegrityStatus.warning,
                        severity="warning",
                        summary={"message": "Table was queued without row-count evidence."},
                        metadata={"runner": "default"},
                    )
                )
            elif isinstance(row_count, int) and row_count >= 0:
                results.append(
                    IntegrityCheckResult(
                        table_name=table,
                        status=IntegrityStatus.pass_,
                        severity="info",
                        summary={"row_count": row_count},
                        metadata={"runner": "default"},
                    )
                )
            else:
                results.append(
                    IntegrityCheckResult(
                        table_name=table,
                        status=IntegrityStatus.fail,
                        severity="error",
                        summary={"message": "Invalid copied row count.", "row_count": row_count},
                        metadata={"runner": "default"},
                    )
                )
        return results


@dataclass
class IntegrityService:
    sqlite: SQLiteStore
    runner: IntegrityRunner | None = None

    def __post_init__(self) -> None:
        if self.runner is None:
            self.runner = DefaultIntegrityRunner()

    def notify_dataset_update(
        self,
        dataset_name: str,
        tables: Iterable[str] | dict[str, Any] | None = None,
        source_version: str | None = None,
        metadata: JsonDict | None = None,
    ) -> ToolResult:
        try:
            dataset_name = self._require_dataset(dataset_name)
            table_names, metadata = self._normalize_tables_and_metadata(tables, metadata)
            source_version = source_version or f"update_{uuid4().hex}"
            now = utc_now_iso()
            job_id = self._new_job_id()
            with self.sqlite.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO integrity_dataset_state(dataset_name, source_version, status, metadata_json, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(dataset_name) DO UPDATE SET
                        source_version = excluded.source_version,
                        status = excluded.status,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                    """,
                    (dataset_name, source_version, IntegrityStatus.stale.value, dumps_json(metadata or {}), now),
                )
                self._mark_tables_stale(conn, dataset_name, table_names, source_version, now, metadata or {})
                conn.execute(
                    """
                    INSERT INTO integrity_queue(
                        job_id, dataset_name, tables_json, source_version, priority, status,
                        requested_by, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        dataset_name,
                        dumps_json(table_names) if table_names is not None else None,
                        source_version,
                        100,
                        IntegrityJobStatus.queued.value,
                        "dataset_update",
                        dumps_json(metadata or {}),
                        now,
                        now,
                    ),
                )
            return ToolResult.success({"dataset_name": dataset_name, "source_version": source_version, "job_id": job_id, "status": "queued"})
        except Exception as exc:
            return ToolResult.from_exception(exc)

    def ensure_tables_ready(
        self,
        dataset_name: str,
        tables: Iterable[str],
        wait_timeout_s: float = 0.0,
        metadata: JsonDict | None = None,
    ) -> ToolResult:
        try:
            dataset_name = self._require_dataset(dataset_name)
            table_names = self._require_tables(tables)
            queued_job_id = self._enqueue_if_needed(dataset_name, table_names, metadata or {})
            deadline = time.monotonic() + max(wait_timeout_s, 0.0)
            while wait_timeout_s > 0 and time.monotonic() <= deadline:
                if not queued_job_id:
                    break
                self.process_next_pending()
                status_by_table = self._status_by_table(dataset_name, table_names)
                if all(status_by_table.get(table, {}).get("status") in READY_STATUSES for table in table_names):
                    break
                if not self._has_pending_for_tables(dataset_name, table_names):
                    break
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            status_by_table = self._status_by_table(dataset_name, table_names)
            ready = all(status_by_table.get(table, {}).get("status") in READY_STATUSES for table in table_names)
            return ToolResult.success(
                {
                    "dataset_name": dataset_name,
                    "tables": table_names,
                    "ready": ready,
                    "queued_job_id": queued_job_id,
                    "status_by_table": status_by_table,
                }
            )
        except Exception as exc:
            return ToolResult.from_exception(exc)

    def get_summary(self, dataset_name: str, tables: Iterable[str] | None = None, include_pass: bool = False) -> ToolResult:
        try:
            dataset_name = self._require_dataset(dataset_name)
            table_names = [str(t) for t in tables] if tables is not None else None
            clauses = ["dataset_name = ?"]
            params: list[Any] = [dataset_name]
            if table_names is not None:
                placeholders = ", ".join("?" for _ in table_names)
                clauses.append(f"(table_name IN ({placeholders}) OR table_name IS NULL)")
                params.extend(table_names)
            rows = self.sqlite.query_all(
                f"""
                SELECT * FROM integrity_summaries
                WHERE {' AND '.join(clauses)}
                ORDER BY table_name, created_at DESC
                """,
                params,
            )
            state = self.sqlite.query_one("SELECT * FROM integrity_dataset_state WHERE dataset_name = ?", (dataset_name,))
            queue = self.sqlite.query_all(
                """
                SELECT job_id, tables_json, source_version, priority, status, requested_by, created_at, updated_at, error_json
                FROM integrity_queue
                WHERE dataset_name = ? AND status IN (?, ?)
                ORDER BY priority ASC, created_at ASC
                """,
                (dataset_name, IntegrityJobStatus.queued.value, IntegrityJobStatus.running.value),
            )
            latest_rows = []
            seen_tables: set[str | None] = set()
            for row in rows:
                table_name = row["table_name"]
                if table_name in seen_tables:
                    continue
                seen_tables.add(table_name)
                latest_rows.append(row)
            latest_rows.sort(
                key=lambda row: (
                    {"fail": 0, "warning": 1, "stale": 2, "pending": 3, "running": 4}.get(row["status"], 5),
                    row["table_name"] or "",
                )
            )

            attention_counts: dict[str, int] = {}
            summaries = []
            for row in latest_rows:
                status = row["status"]
                if status in ATTENTION_STATUSES or include_pass:
                    attention_counts[status] = attention_counts.get(status, 0) + 1
                if include_pass or status != IntegrityStatus.pass_.value:
                    summaries.append(self._summary_row_to_dict(row))
            return ToolResult.success(
                {
                    "dataset_name": dataset_name,
                    "source_version": state["source_version"] if state else None,
                    "status": state["status"] if state else "unknown",
                    "updated_at": state["updated_at"] if state else None,
                    "attention_counts": attention_counts,
                    "summaries": summaries,
                    "pending_jobs": [
                        {
                            "job_id": row["job_id"],
                            "tables": loads_json(row["tables_json"], None) if row["tables_json"] else None,
                            "source_version": row["source_version"],
                            "priority": row["priority"],
                            "status": row["status"],
                            "requested_by": row["requested_by"],
                            "created_at": row["created_at"],
                            "updated_at": row["updated_at"],
                            "error": loads_json(row["error_json"], None) if row["error_json"] else None,
                        }
                        for row in queue
                    ],
                }
            )
        except Exception as exc:
            return ToolResult.from_exception(exc)

    def process_next_pending(self) -> ToolResult:
        try:
            row = self.sqlite.query_one(
                """
                SELECT * FROM integrity_queue
                WHERE status = ?
                ORDER BY priority ASC, created_at ASC
                LIMIT 1
                """,
                (IntegrityJobStatus.queued.value,),
            )
            if row is None:
                return ToolResult.success({"processed": False})
            return self._process_job(row["job_id"])
        except Exception as exc:
            return ToolResult.from_exception(exc)

    def _process_job(self, job_id: str) -> ToolResult:
        row = self.sqlite.query_one("SELECT * FROM integrity_queue WHERE job_id = ?", (job_id,))
        if row is None:
            raise BackendError("NOT_FOUND", "Integrity job was not found.", {"job_id": job_id})
        now = utc_now_iso()
        self.sqlite.execute(
            "UPDATE integrity_queue SET status = ?, started_at = ?, updated_at = ? WHERE job_id = ? AND status = ?",
            (IntegrityJobStatus.running.value, now, now, job_id, IntegrityJobStatus.queued.value),
        )
        dataset_name = row["dataset_name"]
        tables = loads_json(row["tables_json"], None) if row["tables_json"] else None
        metadata = loads_json(row["metadata_json"])
        source_version = row["source_version"]
        try:
            assert self.runner is not None
            results = self.runner.run_checks(dataset_name, tables, source_version, metadata)
            self._persist_results(job_id, dataset_name, source_version, results, metadata)
            completed = utc_now_iso()
            statuses = {result.status.value for result in results}
            queue_status = IntegrityJobStatus.succeeded.value if statuses and statuses.issubset(SUCCESS_STATUSES) else IntegrityJobStatus.failed.value
            error_json = None
            if queue_status == IntegrityJobStatus.failed.value:
                error_json = dumps_json({
                    "message": "One or more integrity checks require attention.",
                    "statuses": sorted(statuses),
                })
            self.sqlite.execute(
                "UPDATE integrity_queue SET status = ?, completed_at = ?, updated_at = ?, error_json = ? WHERE job_id = ?",
                (queue_status, completed, completed, error_json, job_id),
            )
            self._refresh_dataset_status(dataset_name)
            return ToolResult.success({"processed": True, "job_id": job_id, "result_count": len(results), "job_status": queue_status})
        except Exception as exc:
            failed = utc_now_iso()
            self.sqlite.execute(
                "UPDATE integrity_queue SET status = ?, completed_at = ?, updated_at = ?, error_json = ? WHERE job_id = ?",
                (IntegrityJobStatus.failed.value, failed, failed, dumps_json({"type": type(exc).__name__, "message": str(exc)}), job_id),
            )
            self._refresh_dataset_status(dataset_name)
            return ToolResult.from_exception(exc)

    def _persist_results(
        self,
        job_id: str,
        dataset_name: str,
        source_version: str | None,
        results: list[IntegrityCheckResult],
        job_metadata: JsonDict,
    ) -> None:
        now = utc_now_iso()
        with self.sqlite.connect() as conn:
            for result in results:
                conn.execute(
                    """
                    INSERT INTO integrity_summaries(
                        summary_id, job_id, dataset_name, table_name, source_version, status,
                        severity, summary_json, artifact_refs_json, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self._new_summary_id(),
                        job_id,
                        dataset_name,
                        result.table_name,
                        source_version,
                        result.status.value,
                        result.severity,
                        dumps_json(result.summary),
                        dumps_json(result.artifact_refs),
                        dumps_json({**job_metadata, **result.metadata}),
                        now,
                    ),
                )

    def _enqueue_if_needed(self, dataset_name: str, tables: list[str], metadata: JsonDict) -> str | None:
        status_by_table = self._status_by_table(dataset_name, tables)
        if all(status_by_table.get(table, {}).get("status") in READY_STATUSES for table in tables):
            return None
        pending_rows = self.sqlite.query_all(
            """
            SELECT job_id, tables_json FROM integrity_queue
            WHERE dataset_name = ? AND status IN (?, ?)
            ORDER BY priority ASC, created_at ASC
            """,
            (dataset_name, IntegrityJobStatus.queued.value, IntegrityJobStatus.running.value),
        )
        requested = set(tables)
        for row in pending_rows:
            queued_tables = loads_json(row["tables_json"], None) if row["tables_json"] else None
            if queued_tables is None or requested.issubset(set(queued_tables)):
                return row["job_id"]
        state = self.sqlite.query_one("SELECT source_version FROM integrity_dataset_state WHERE dataset_name = ?", (dataset_name,))
        source_version = state["source_version"] if state else f"query_{uuid4().hex}"
        job_id = self._new_job_id()
        now = utc_now_iso()
        with self.sqlite.connect() as conn:
            if state is None:
                conn.execute(
                    """
                    INSERT INTO integrity_dataset_state(dataset_name, source_version, status, metadata_json, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (dataset_name, source_version, IntegrityStatus.pending.value, dumps_json(metadata), now),
                )
            self._mark_pending_tables(conn, dataset_name, tables, source_version, now, metadata)
            conn.execute(
                """
                INSERT INTO integrity_queue(job_id, dataset_name, tables_json, source_version, priority, status, requested_by, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, dataset_name, dumps_json(tables), source_version, 10, IntegrityJobStatus.queued.value, "ensure_tables_ready", dumps_json(metadata), now, now),
            )
        return job_id

    def _status_by_table(self, dataset_name: str, tables: list[str]) -> dict[str, JsonDict]:
        status: dict[str, JsonDict] = {}
        for table in tables:
            row = self.sqlite.query_one(
                """
                SELECT * FROM integrity_summaries
                WHERE dataset_name = ? AND table_name = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (dataset_name, table),
            )
            if row is None:
                status[table] = {"status": IntegrityStatus.pending.value, "summary": {"message": "No integrity result recorded."}}
            else:
                status[table] = self._summary_row_to_dict(row)
        return status

    def _has_pending_for_tables(self, dataset_name: str, tables: list[str]) -> bool:
        rows = self.sqlite.query_all(
            "SELECT tables_json FROM integrity_queue WHERE dataset_name = ? AND status IN (?, ?)",
            (dataset_name, IntegrityJobStatus.queued.value, IntegrityJobStatus.running.value),
        )
        requested = set(tables)
        for row in rows:
            queued_tables = loads_json(row["tables_json"], None) if row["tables_json"] else None
            if queued_tables is None or requested.intersection(queued_tables):
                return True
        return False

    def _mark_pending_tables(self, conn, dataset_name: str, tables: list[str], source_version: str, now: str, metadata: JsonDict) -> None:
        for table in tables:
            latest = conn.execute(
                """
                SELECT status, source_version FROM integrity_summaries
                WHERE dataset_name = ? AND table_name = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (dataset_name, table),
            ).fetchone()
            if latest is not None and latest[0] in READY_STATUSES and latest[1] == source_version:
                continue
            conn.execute(
                """
                INSERT INTO integrity_summaries(
                    summary_id, job_id, dataset_name, table_name, source_version, status, severity,
                    summary_json, artifact_refs_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._new_summary_id(),
                    None,
                    dataset_name,
                    table,
                    source_version,
                    IntegrityStatus.pending.value,
                    "info",
                    dumps_json({"message": "Integrity check was prioritized for this table and is pending."}),
                    dumps_json([]),
                    dumps_json(metadata),
                    now,
                ),
            )

    def _mark_tables_stale(self, conn, dataset_name: str, tables: list[str] | None, source_version: str, now: str, metadata: JsonDict) -> None:
        if tables is None:
            existing = self.sqlite.query_all("SELECT DISTINCT table_name FROM integrity_summaries WHERE dataset_name = ? AND table_name IS NOT NULL", (dataset_name,))
            tables = [row["table_name"] for row in existing]
        for table in tables or []:
            conn.execute(
                """
                INSERT INTO integrity_summaries(
                    summary_id, job_id, dataset_name, table_name, source_version, status, severity,
                    summary_json, artifact_refs_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._new_summary_id(),
                    None,
                    dataset_name,
                    table,
                    source_version,
                    IntegrityStatus.stale.value,
                    "warning",
                    dumps_json({"message": "Dataset was updated; integrity check is pending."}),
                    dumps_json([]),
                    dumps_json(metadata),
                    now,
                ),
            )

    def _refresh_dataset_status(self, dataset_name: str) -> None:
        rows = self.sqlite.query_all(
            """
            SELECT table_name, status FROM integrity_summaries
            WHERE dataset_name = ?
            ORDER BY table_name, created_at DESC
            """,
            (dataset_name,),
        )
        latest_by_table: dict[str | None, str] = {}
        for row in rows:
            table_name = row["table_name"]
            if table_name not in latest_by_table:
                latest_by_table[table_name] = row["status"]
        statuses = list(latest_by_table.values())
        if any(status == IntegrityStatus.fail.value for status in statuses):
            status = IntegrityStatus.fail.value
        elif any(status in {IntegrityStatus.warning.value, IntegrityStatus.stale.value} for status in statuses):
            status = IntegrityStatus.warning.value
        elif statuses:
            status = IntegrityStatus.pass_.value
        else:
            status = IntegrityStatus.pending.value
        self.sqlite.execute("UPDATE integrity_dataset_state SET status = ?, updated_at = ? WHERE dataset_name = ?", (status, utc_now_iso(), dataset_name))

    def _summary_row_to_dict(self, row) -> JsonDict:
        return {
            "summary_id": row["summary_id"],
            "job_id": row["job_id"],
            "dataset_name": row["dataset_name"],
            "table_name": row["table_name"],
            "source_version": row["source_version"],
            "status": row["status"],
            "severity": row["severity"],
            "summary": loads_json(row["summary_json"]),
            "artifact_refs": loads_json(row["artifact_refs_json"], []),
            "metadata": loads_json(row["metadata_json"]),
            "created_at": row["created_at"],
        }

    def _normalize_tables_and_metadata(self, tables: Iterable[str] | dict[str, Any] | None, metadata: JsonDict | None) -> tuple[list[str] | None, JsonDict]:
        metadata = dict(metadata or {})
        if isinstance(tables, dict):
            metadata.setdefault("row_counts", dict(tables))
            return [str(table) for table in tables.keys()], metadata
        if tables is None:
            return None, metadata
        return [str(table) for table in tables], metadata

    def _require_dataset(self, dataset_name: str) -> str:
        dataset_name = str(dataset_name).strip()
        if not dataset_name:
            raise BackendError("VALIDATION_ERROR", "dataset_name is required.")
        return dataset_name

    def _require_tables(self, tables: Iterable[str]) -> list[str]:
        table_names = [str(table).strip() for table in tables if str(table).strip()]
        if not table_names:
            raise BackendError("VALIDATION_ERROR", "At least one table is required.")
        return table_names

    def _new_job_id(self) -> str:
        return f"intjob_{uuid4().hex}"

    def _new_summary_id(self) -> str:
        return f"intsum_{uuid4().hex}"

from __future__ import annotations

import uuid
from typing import Optional

from data_agent_backend.models.common import BackendError, utc_now_iso
from data_agent_backend.models.datasource import (
    CatalogSummary,
    DatasourceCreateRequest,
    DatasourceRecord,
)
from data_agent_backend.services.connectors.mysql_connector import MySQLConnector
from data_agent_backend.storage.sqlite import SQLiteStore, dumps_json, loads_json


class DatasourceService:
    """Manages datasource registration, deduplication, query execution, and catalog."""

    def __init__(self, sqlite: SQLiteStore) -> None:
        self._sqlite = sqlite
        self._passwords: dict[str, str] = {}
        self._catalog_cache: dict[str, CatalogSummary] = {}

    # ── Registration ──

    def register(self, req: DatasourceCreateRequest) -> DatasourceRecord:
        existing = self._find_matching(req)
        if existing:
            self._passwords[existing.datasource_id] = req.password
            return existing

        ds_id = f"ds_{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()
        self._sqlite.execute(
            """
            INSERT INTO datasources(datasource_id, name, type, host, port, database_name, username, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ds_id, req.name, req.type, req.host, req.port, req.database, req.username, dumps_json(req.metadata), now),
        )
        self._passwords[ds_id] = req.password
        return DatasourceRecord(
            datasource_id=ds_id,
            name=req.name,
            type=req.type,
            host=req.host,
            port=req.port,
            database=req.database,
            username=req.username,
            created_at=now,
            metadata=req.metadata,
        )

    def get(self, datasource_id: str) -> DatasourceRecord:
        row = self._sqlite.query_one("SELECT * FROM datasources WHERE datasource_id = ?", (datasource_id,))
        if not row:
            raise BackendError("NOT_FOUND", f"Datasource {datasource_id} not found.")
        return self._row_to_record(row)

    def list_all(self) -> list[DatasourceRecord]:
        rows = self._sqlite.query_all("SELECT * FROM datasources ORDER BY created_at")
        return [self._row_to_record(r) for r in rows]

    def get_default_id(self) -> Optional[str]:
        row = self._sqlite.query_one("SELECT datasource_id FROM datasources ORDER BY created_at LIMIT 1")
        return row["datasource_id"] if row else None

    # ── Query execution ──

    def query_datasource(
        self, datasource_id: str, query: str, row_limit: int = 1000,
    ) -> tuple[list[tuple], list[str], str]:
        """Execute a read-only query and return (rows, columns, csv_text)."""
        record = self.get(datasource_id)
        connector = self._get_connector(record)
        rows, columns = connector.execute_query(query, row_limit)
        csv_text = connector.rows_to_csv(columns, rows)
        return rows, columns, csv_text

    # ── Catalog ──

    def get_catalog_summary(self, datasource_id: str) -> Optional[CatalogSummary]:
        return self._catalog_cache.get(datasource_id)

    def refresh_catalog(self, datasource_id: str) -> CatalogSummary:
        record = self.get(datasource_id)
        connector = self._get_connector(record)
        catalog = connector.fetch_catalog()
        self._catalog_cache[datasource_id] = catalog
        return catalog

    # ── Internals ──

    def _find_matching(self, req: DatasourceCreateRequest) -> Optional[DatasourceRecord]:
        row = self._sqlite.query_one(
            "SELECT * FROM datasources WHERE name = ? AND host = ? AND database_name = ? AND username = ?",
            (req.name, req.host, req.database, req.username),
        )
        if row:
            record = self._row_to_record(row)
            return record
        return None

    def _get_connector(self, record: DatasourceRecord) -> MySQLConnector:
        password = self._passwords.get(record.datasource_id, "")
        return MySQLConnector(record, password)

    def set_password(self, datasource_id: str, password: str) -> None:
        self._passwords[datasource_id] = password

    @staticmethod
    def _row_to_record(row) -> DatasourceRecord:
        return DatasourceRecord(
            datasource_id=row["datasource_id"],
            name=row["name"],
            type=row["type"],
            host=row["host"],
            port=row["port"],
            database=row["database_name"],
            username=row["username"],
            created_at=row["created_at"],
            metadata=loads_json(row["metadata_json"]),
        )

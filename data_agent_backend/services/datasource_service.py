from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from data_agent_backend.models.common import BackendError, utc_now_iso
from data_agent_backend.models.db_tools import (
    AgentDatasourceSummary,
    DBColumnSummary,
    DBRowsPreview,
    DBSchemaSummary,
    DBTableDescription,
    DBTableSummary,
)
from data_agent_backend.models.datasource import (
    CatalogSummary,
    DatasourceCredential,
    DatasourceCreateRequest,
    DatasourceRecord,
)
from data_agent_backend.services.connectors.base import DatasourceConnector
from data_agent_backend.services.connectors.registry import ConnectorRegistry, default_connector_registry
from data_agent_backend.storage.sqlite import SQLiteStore, dumps_json, loads_json


class DatasourceService:
    """Manages datasource metadata and per-request connector operations."""

    def __init__(self, sqlite: SQLiteStore, connector_registry: ConnectorRegistry | None = None) -> None:
        self._sqlite = sqlite
        self._connector_registry = connector_registry or default_connector_registry
        self._catalog_cache: dict[str, CatalogSummary] = {}

    def register(self, req: DatasourceCreateRequest) -> DatasourceRecord:
        existing = self._find_matching(req)
        if existing:
            return existing

        ds_id = f"ds_{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()
        metadata = dict(req.metadata)
        if req.path is not None:
            metadata["path"] = req.path
        host = req.host or ""
        port = 0 if req.port is None else req.port
        database = req.database or req.path or ""
        username = req.username or ""
        self._sqlite.execute(
            """
            INSERT INTO datasources(datasource_id, name, type, host, port, database_name, username, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ds_id, req.name, req.type.value, host, port, database, username, dumps_json(metadata), now),
        )
        return DatasourceRecord(
            datasource_id=ds_id,
            name=req.name,
            type=req.type,
            host=req.host,
            port=req.port,
            database=req.database or req.path,
            username=req.username,
            path=req.path,
            created_at=now,
            metadata=metadata,
        )

    def get(self, datasource_id: str) -> DatasourceRecord:
        row = self._sqlite.query_one("SELECT * FROM datasources WHERE datasource_id = ?", (datasource_id,))
        if not row:
            raise BackendError("NOT_FOUND", f"Datasource {datasource_id} not found.")
        return self._row_to_record(row)

    def list_all(self) -> list[DatasourceRecord]:
        rows = self._sqlite.query_all("SELECT * FROM datasources ORDER BY created_at")
        return [self._row_to_record(r) for r in rows]

    def resolve_datasource_id(self, datasource_id: str | None) -> str:
        if datasource_id is None:
            raise BackendError("DATASOURCE_ID_REQUIRED", "datasource_id is required.")
        self.get(datasource_id)
        return datasource_id

    def list_agent_datasources(self) -> list[AgentDatasourceSummary]:
        return [
            AgentDatasourceSummary(
                datasource_id=item.datasource_id,
                name=item.name,
                type=item.type.value if hasattr(item.type, "value") else str(item.type),
                database=self._safe_database_label(item),
                is_default=False,
            )
            for item in self.list_all()
        ]

    def test_connection(self, datasource_id: str, credential: DatasourceCredential | None = None) -> bool:
        connector = self._get_connector(self.get(datasource_id), credential)
        try:
            return connector.test_connection()
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(
                "DATASOURCE_CONNECTION_FAILED",
                "Datasource connection failed.",
                {"datasource_id": datasource_id, "type": type(exc).__name__},
            ) from exc

    def get_schema(self, datasource_id: str | None = None, credential: DatasourceCredential | None = None) -> DBSchemaSummary:
        resolved_id = self.resolve_datasource_id(datasource_id)
        record = self.get(resolved_id)
        connector = self._get_connector(record, credential)
        catalog = connector.fetch_catalog()
        return DBSchemaSummary(
            datasource_id=resolved_id,
            database=record.database or "",
            tables=[
                DBTableSummary(
                    name=table_name,
                    columns=[
                        DBColumnSummary(
                            name=column_name,
                            type=column.type,
                            nullable=column.nullable,
                            key=column.key,
                            description=column.description,
                        )
                        for column_name, column in table.columns.items()
                    ],
                )
                for table_name, table in catalog.tables.items()
            ],
        )

    def describe_table(
        self,
        datasource_id: str | None,
        table_name: str,
        credential: DatasourceCredential | None = None,
    ) -> DBTableDescription:
        resolved_id = self.resolve_datasource_id(datasource_id)
        record = self.get(resolved_id)
        connector = self._get_connector(record, credential)
        table = connector.describe_table(table_name)
        return DBTableDescription(
            datasource_id=resolved_id,
            database=record.database or "",
            table_name=table_name,
            columns=[
                DBColumnSummary(
                    name=column_name,
                    type=column.type,
                    nullable=column.nullable,
                    key=column.key,
                    description=column.description,
                )
                for column_name, column in table.columns.items()
            ],
        )

    def sample_rows(
        self,
        datasource_id: str | None,
        table_name: str,
        limit: int | None = None,
        credential: DatasourceCredential | None = None,
    ) -> DBRowsPreview:
        resolved_id = self.resolve_datasource_id(datasource_id)
        record = self.get(resolved_id)
        safe_limit = 50 if limit is None else int(limit)
        if safe_limit <= 0:
            raise BackendError("POLICY_BLOCKED", "row_limit must be positive.")
        connector = self._get_connector(record, credential)
        rows, columns = connector.sample_rows(table_name, safe_limit)
        return DBRowsPreview(
            datasource_id=resolved_id,
            table_name=table_name,
            columns=columns,
            rows=[dict(zip(columns, row)) for row in rows],
            limit=safe_limit,
            truncated=False,
        )

    def query_datasource(
        self,
        datasource_id: str,
        credential: DatasourceCredential | None,
        query: str,
        row_limit: int = 1000,
    ) -> tuple[list[tuple], list[str], str]:
        """Execute a read-only query and return (rows, columns, csv_text)."""
        record = self.get(datasource_id)
        connector = self._get_connector(record, credential)
        try:
            rows, columns = connector.execute_query(query, row_limit)
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(
                "DATASOURCE_QUERY_FAILED",
                "Datasource query failed.",
                {"datasource_id": datasource_id, "type": type(exc).__name__},
            ) from exc
        csv_text = connector.rows_to_csv(columns, rows)
        return rows, columns, csv_text

    def get_catalog_summary(self, datasource_id: str) -> Optional[CatalogSummary]:
        return self._catalog_cache.get(datasource_id)

    def refresh_catalog(self, datasource_id: str, credential: DatasourceCredential | None = None) -> CatalogSummary:
        record = self.get(datasource_id)
        connector = self._get_connector(record, credential)
        catalog = connector.fetch_catalog()
        self._catalog_cache[datasource_id] = catalog
        return catalog

    def _find_matching(self, req: DatasourceCreateRequest) -> Optional[DatasourceRecord]:
        row = self._sqlite.query_one(
            "SELECT * FROM datasources WHERE name = ? AND type = ? AND host = ? AND database_name = ? AND username = ?",
            (req.name, req.type.value, req.host or "", req.database or req.path or "", req.username or ""),
        )
        return self._row_to_record(row) if row else None

    def _get_connector(self, record: DatasourceRecord, credential: DatasourceCredential | None = None) -> DatasourceConnector:
        if credential is None:
            raise BackendError("CREDENTIAL_REQUIRED", "Datasource credential is required.", {"datasource_id": record.datasource_id})
        return self._connector_registry.create(record, credential)

    @staticmethod
    def _safe_database_label(record: DatasourceRecord) -> str:
        if record.path:
            return Path(record.path).name
        return record.database or ""

    @staticmethod
    def _row_to_record(row) -> DatasourceRecord:
        metadata = loads_json(row["metadata_json"])
        return DatasourceRecord(
            datasource_id=row["datasource_id"],
            name=row["name"],
            type=row["type"],
            host=row["host"] or None,
            port=row["port"] or None,
            database=row["database_name"] or None,
            username=row["username"] or None,
            path=metadata.get("path"),
            created_at=row["created_at"],
            metadata=metadata,
        )

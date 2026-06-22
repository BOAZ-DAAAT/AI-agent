from __future__ import annotations

import csv
import io
import re
import sqlite3
from pathlib import Path

from data_agent_backend.models.common import BackendError, utc_now_iso
from data_agent_backend.models.datasource import CatalogSummary, ColumnInfo, DatasourceCredential, DatasourceRecord, TableSummary


SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SQLiteDatasourceConnector:
    """Connector for user-provided analytical SQLite databases."""

    def __init__(self, record: DatasourceRecord, credential: DatasourceCredential) -> None:
        self._record = record
        self._credential = credential

    def _path(self) -> Path:
        if not self._record.path:
            raise BackendError("INVALID_DATASOURCE_CONFIG", "SQLite datasource path is required.", {"datasource_id": self._record.datasource_id})
        return Path(self._record.path)

    def _connect(self) -> sqlite3.Connection:
        path = self._path()
        if not path.exists():
            raise BackendError("DATASOURCE_CONNECTION_FAILED", "SQLite datasource file was not found.", {"datasource_id": self._record.datasource_id})
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def test_connection(self) -> bool:
        conn = self._connect()
        try:
            conn.execute("SELECT 1").fetchone()
            return True
        finally:
            conn.close()

    def quote_identifier(self, name: str) -> str:
        if not SAFE_IDENTIFIER_RE.fullmatch(name):
            raise BackendError("INVALID_IDENTIFIER", "Only simple SQLite identifiers are allowed.", {"identifier": name})
        return f'"{name}"'

    def execute_query(self, query: str, row_limit: int = 1000) -> tuple[list[tuple], list[str]]:
        wrapped = f"SELECT * FROM ({query.rstrip(';')}) AS _daa_sub LIMIT ?"
        conn = self._connect()
        try:
            cur = conn.execute(wrapped, (int(row_limit),))
            rows = [tuple(row) for row in cur.fetchall()]
            columns = [desc[0] for desc in cur.description or []]
            return rows, columns
        finally:
            conn.close()

    def describe_table(self, table_name: str) -> TableSummary:
        quoted = self.quote_identifier(table_name)
        conn = self._connect()
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ? LIMIT 1",
                (table_name,),
            ).fetchone()
            if exists is None:
                raise BackendError("NOT_FOUND", f"Table {table_name} was not found.", {"table_name": table_name})
            rows = conn.execute(f"PRAGMA table_info({quoted})").fetchall()
            columns = {
                row["name"]: ColumnInfo(type=row["type"] or "", nullable=not bool(row["notnull"]), key="PRI" if row["pk"] else "")
                for row in rows
            }
            return TableSummary(columns=columns)
        finally:
            conn.close()

    def sample_rows(self, table_name: str, limit: int) -> tuple[list[tuple], list[str]]:
        quoted = self.quote_identifier(table_name)
        conn = self._connect()
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ? LIMIT 1",
                (table_name,),
            ).fetchone()
            if exists is None:
                raise BackendError("NOT_FOUND", f"Table {table_name} was not found.", {"table_name": table_name})
            cur = conn.execute(f"SELECT * FROM {quoted} LIMIT ?", (int(limit),))
            rows = [tuple(row) for row in cur.fetchall()]
            columns = [desc[0] for desc in cur.description or []]
            return rows, columns
        finally:
            conn.close()

    def fetch_catalog(self) -> CatalogSummary:
        conn = self._connect()
        try:
            table_names = [
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
                ).fetchall()
            ]
            tables = {table_name: self.describe_table(table_name) for table_name in table_names}
            return CatalogSummary(
                datasource_id=self._record.datasource_id,
                database=self._record.database,
                tables=tables,
                refreshed_at=utc_now_iso(),
            )
        finally:
            conn.close()

    def rows_to_csv(self, columns: list[str], rows: list[tuple]) -> str:
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(columns)
        writer.writerows(rows)
        return out.getvalue()

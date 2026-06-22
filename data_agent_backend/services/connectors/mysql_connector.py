from __future__ import annotations

import csv
import io
from typing import Any

try:
    import pymysql
except ModuleNotFoundError:  # pragma: no cover - depends on optional runtime package
    pymysql = None

from data_agent_backend.models.common import BackendError, utc_now_iso
from data_agent_backend.models.datasource import (
    CatalogSummary,
    ColumnInfo,
    DatasourceRecord,
    TableSummary,
)


class MySQLConnector:
    """Encapsulates MySQL connection, query execution, and catalog introspection."""

    def __init__(self, record: DatasourceRecord, password: str = "") -> None:
        self._record = record
        self._password = password

    def _connect(self) -> pymysql.Connection:
        if pymysql is None:
            raise BackendError(
                "DEPENDENCY_MISSING",
                "pymysql is required for MySQL datasource connections.",
                {"package": "pymysql"},
            )
        return pymysql.connect(
            host=self._record.host,
            port=self._record.port,
            user=self._record.username,
            password=self._password,
            database=self._record.database,
            charset="utf8mb4",
            connect_timeout=10,
            read_timeout=30,
        )

    def test_connection(self) -> bool:
        try:
            conn = self._connect()
            conn.close()
            return True
        except Exception:
            return False

    def execute_query(self, query: str, row_limit: int = 1000) -> tuple[list[tuple], list[str]]:
        wrapped = f"SELECT * FROM ({query.rstrip(';')}) AS _daa_sub LIMIT {int(row_limit)}"
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(wrapped)
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description or []]
                return list(rows), columns
        finally:
            conn.close()

    def fetch_catalog(self) -> CatalogSummary:
        conn = self._connect()
        try:
            tables: dict[str, TableSummary] = {}
            with conn.cursor() as cur:
                cur.execute("SHOW TABLES")
                table_names = [row[0] for row in cur.fetchall()]

            for tname in table_names:
                with conn.cursor() as cur:
                    cur.execute(f"DESCRIBE `{tname}`")
                    col_rows = cur.fetchall()
                columns: dict[str, ColumnInfo] = {}
                for col in col_rows:
                    columns[col[0]] = ColumnInfo(
                        type=col[1],
                        nullable=(col[2] == "YES"),
                        key=col[3] or "",
                    )
                tables[tname] = TableSummary(columns=columns)

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

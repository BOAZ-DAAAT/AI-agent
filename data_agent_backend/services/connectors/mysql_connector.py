from __future__ import annotations

import csv
import io
import re
from typing import Any

try:
    import pymysql
except ModuleNotFoundError:  # pragma: no cover - depends on optional runtime package
    pymysql = None

from data_agent_backend.models.common import BackendError, utc_now_iso
from data_agent_backend.models.datasource import (
    CatalogSummary,
    ColumnInfo,
    DatasourceCredential,
    DatasourceRecord,
    TableSummary,
)


SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class MySQLConnector:
    """Encapsulates MySQL connection, query execution, and catalog introspection."""

    def __init__(self, record: DatasourceRecord, credential: DatasourceCredential | str | None = None) -> None:
        self._record = record
        if isinstance(credential, str):
            credential = DatasourceCredential(password=credential)
        self._credential = credential or DatasourceCredential()

    def _require_password(self) -> str:
        if not self._credential.password:
            raise BackendError("CREDENTIAL_REQUIRED", "Datasource password is required for MySQL connections.", {"datasource_id": self._record.datasource_id})
        return self._credential.password

    def _connect(self) -> pymysql.Connection:
        password = self._require_password()
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
            password=password,
            database=self._record.database,
            charset="utf8mb4",
            connect_timeout=10,
            read_timeout=30,
        )

    def test_connection(self) -> bool:
        conn = self._connect()
        conn.close()
        return True

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

    def quote_identifier(self, name: str) -> str:
        if not SAFE_IDENTIFIER_RE.fullmatch(name):
            raise BackendError(
                "INVALID_IDENTIFIER",
                "Only simple MySQL identifiers are allowed.",
                {"identifier": name},
            )
        return f"`{name}`"

    def table_exists(self, table_name: str) -> bool:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = %s
                    LIMIT 1
                    """,
                    (self._record.database, table_name),
                )
                return cur.fetchone() is not None
        finally:
            conn.close()

    def describe_table(self, table_name: str) -> TableSummary:
        quoted = self.quote_identifier(table_name)
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = %s
                    LIMIT 1
                    """,
                    (self._record.database, table_name),
                )
                if cur.fetchone() is None:
                    raise BackendError(
                        "NOT_FOUND",
                        f"Table {table_name} was not found.",
                        {"table_name": table_name},
                    )

            with conn.cursor() as cur:
                cur.execute(f"DESCRIBE {quoted}")
                col_rows = cur.fetchall()

            columns: dict[str, ColumnInfo] = {}
            for col in col_rows:
                columns[col[0]] = ColumnInfo(
                    type=col[1],
                    nullable=(col[2] == "YES"),
                    key=col[3] or "",
                )
            return TableSummary(columns=columns)
        finally:
            conn.close()

    def sample_rows(self, table_name: str, limit: int) -> tuple[list[tuple], list[str]]:
        quoted = self.quote_identifier(table_name)
        safe_limit = max(1, int(limit))
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = %s
                    LIMIT 1
                    """,
                    (self._record.database, table_name),
                )
                if cur.fetchone() is None:
                    raise BackendError(
                        "NOT_FOUND",
                        f"Table {table_name} was not found.",
                        {"table_name": table_name},
                    )

            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {quoted} LIMIT %s", (safe_limit,))
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
                    cur.execute(f"DESCRIBE {self.quote_identifier(tname)}")
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

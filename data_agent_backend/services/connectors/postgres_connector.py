from __future__ import annotations

import csv
import io
import re

try:
    import psycopg
except ModuleNotFoundError:  # pragma: no cover - optional runtime package
    psycopg = None

try:
    import psycopg2
except ModuleNotFoundError:  # pragma: no cover - optional runtime package
    psycopg2 = None

from data_agent_backend.models.common import BackendError, utc_now_iso
from data_agent_backend.models.datasource import CatalogSummary, ColumnInfo, DatasourceCredential, DatasourceRecord, TableSummary


SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PostgreSQLConnector:
    def __init__(self, record: DatasourceRecord, credential: DatasourceCredential) -> None:
        self._record = record
        self._credential = credential

    def _require_password(self) -> str:
        if not self._credential.password:
            raise BackendError("CREDENTIAL_REQUIRED", "Datasource password is required for PostgreSQL connections.", {"datasource_id": self._record.datasource_id})
        return self._credential.password

    def _connect(self):
        password = self._require_password()
        if psycopg is not None:
            return psycopg.connect(
                host=self._record.host,
                port=self._record.port,
                dbname=self._record.database,
                user=self._record.username,
                password=password,
                connect_timeout=10,
            )
        if psycopg2 is not None:
            return psycopg2.connect(
                host=self._record.host,
                port=self._record.port,
                dbname=self._record.database,
                user=self._record.username,
                password=password,
                connect_timeout=10,
            )
        raise BackendError("DEPENDENCY_MISSING", "psycopg or psycopg2 is required for PostgreSQL datasource connections.", {"package": "psycopg"})

    def test_connection(self) -> bool:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        finally:
            conn.close()

    def quote_identifier(self, name: str) -> str:
        if not SAFE_IDENTIFIER_RE.fullmatch(name):
            raise BackendError("INVALID_IDENTIFIER", "Only simple PostgreSQL identifiers are allowed.", {"identifier": name})
        return f'"{name}"'

    def execute_query(self, query: str, row_limit: int = 1000) -> tuple[list[tuple], list[str]]:
        wrapped = f"SELECT * FROM ({query.rstrip(';')}) AS _daa_sub LIMIT %s"
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(wrapped, (int(row_limit),))
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description or []]
                return list(rows), columns
        finally:
            conn.close()

    def describe_table(self, table_name: str) -> TableSummary:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (table_name,),
                )
                rows = cur.fetchall()
            if not rows:
                raise BackendError("NOT_FOUND", f"Table {table_name} was not found.", {"table_name": table_name})
            columns = {row[0]: ColumnInfo(type=row[1], nullable=(row[2] == "YES")) for row in rows}
            return TableSummary(columns=columns)
        finally:
            conn.close()

    def sample_rows(self, table_name: str, limit: int) -> tuple[list[tuple], list[str]]:
        quoted = self.quote_identifier(table_name)
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {quoted} LIMIT %s", (int(limit),))
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description or []]
                return list(rows), columns
        finally:
            conn.close()

    def fetch_catalog(self) -> CatalogSummary:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                    """
                )
                table_names = [row[0] for row in cur.fetchall()]
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

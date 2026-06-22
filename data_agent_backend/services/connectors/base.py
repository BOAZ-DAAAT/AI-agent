from __future__ import annotations

from typing import Protocol

from data_agent_backend.models.datasource import CatalogSummary, TableSummary


class DatasourceConnector(Protocol):
    def test_connection(self) -> bool:
        ...

    def fetch_catalog(self) -> CatalogSummary:
        ...

    def describe_table(self, table_name: str) -> TableSummary:
        ...

    def sample_rows(self, table_name: str, limit: int) -> tuple[list[tuple], list[str]]:
        ...

    def execute_query(self, query: str, row_limit: int = 1000) -> tuple[list[tuple], list[str]]:
        ...

    def rows_to_csv(self, columns: list[str], rows: list[tuple]) -> str:
        ...

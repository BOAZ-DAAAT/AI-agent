from __future__ import annotations

from .common import BackendModel, JsonDict, StrEnum


class DatasourceType(StrEnum):
    mysql = "mysql"
    duckdb = "duckdb"


class DatasourceCreateRequest(BackendModel):
    name: str
    type: DatasourceType
    host: str
    port: int = 3306
    database: str
    username: str
    password: str = ""
    metadata: JsonDict = {}


class DatasourceRecord(BackendModel):
    datasource_id: str
    name: str
    type: DatasourceType
    host: str
    port: int
    database: str
    username: str
    created_at: str
    metadata: JsonDict = {}

    def matches(self, req: DatasourceCreateRequest) -> bool:
        """Check if this record matches a create request (dedup key)."""
        return (
            self.name == req.name
            and self.host == req.host
            and self.database == req.database
            and self.username == req.username
        )


class ColumnInfo(BackendModel):
    type: str
    nullable: bool = True
    key: str = ""
    description: str = ""


class TableSummary(BackendModel):
    columns: dict[str, ColumnInfo] = {}
    row_count: int | None = None


class CatalogSummary(BackendModel):
    datasource_id: str
    database: str
    tables: dict[str, TableSummary] = {}
    refreshed_at: str = ""

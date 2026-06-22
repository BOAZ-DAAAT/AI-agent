from __future__ import annotations

from pydantic import Field, model_validator

from .common import BackendModel, JsonDict, StrEnum


class DatasourceType(StrEnum):
    mysql = "mysql"
    postgresql = "postgresql"
    sqlite = "sqlite"


class DatasourceCreateRequest(BackendModel):
    name: str
    type: DatasourceType
    host: str | None = None
    port: int | None = None
    database: str | None = None
    username: str | None = None
    path: str | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_by_type(self) -> "DatasourceCreateRequest":
        if self.type in {DatasourceType.mysql, DatasourceType.postgresql}:
            missing = [name for name in ("host", "port", "database", "username") if getattr(self, name) in (None, "")]
            if missing:
                raise ValueError(f"{self.type} datasource requires: {', '.join(missing)}")
        if self.type == DatasourceType.sqlite and not self.path:
            raise ValueError("sqlite datasource requires: path")
        return self


class DatasourceCredential(BackendModel):
    password: str | None = None


class DatasourceRecord(BackendModel):
    datasource_id: str
    name: str
    type: DatasourceType
    host: str | None = None
    port: int | None = None
    database: str | None = None
    username: str | None = None
    path: str | None = None
    created_at: str
    metadata: JsonDict = Field(default_factory=dict)

    def matches(self, req: DatasourceCreateRequest) -> bool:
        """Check if this record matches a create request (dedup key)."""
        return (
            self.name == req.name
            and self.type == req.type
            and self.host == req.host
            and self.database == req.database
            and self.username == req.username
            and self.path == req.path
        )


class ColumnInfo(BackendModel):
    type: str
    nullable: bool = True
    key: str = ""
    description: str = ""


class TableSummary(BackendModel):
    columns: dict[str, ColumnInfo] = Field(default_factory=dict)
    row_count: int | None = None


class CatalogSummary(BackendModel):
    datasource_id: str
    database: str | None = None
    tables: dict[str, TableSummary] = Field(default_factory=dict)
    refreshed_at: str = ""

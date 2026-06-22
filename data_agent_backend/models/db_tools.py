from __future__ import annotations

from data_agent_backend.models.common import BackendModel, JsonDict


class AgentDatasourceSummary(BackendModel):
    datasource_id: str
    name: str
    type: str
    database: str
    is_default: bool = False


class DBColumnSummary(BackendModel):
    name: str
    type: str
    nullable: bool = True
    key: str = ""
    description: str = ""


class DBTableSummary(BackendModel):
    name: str
    columns: list[DBColumnSummary]


class DBSchemaSummary(BackendModel):
    datasource_id: str
    database: str
    tables: list[DBTableSummary]


class DBTableDescription(BackendModel):
    datasource_id: str
    database: str
    table_name: str
    columns: list[DBColumnSummary]


class DBRowsPreview(BackendModel):
    datasource_id: str
    table_name: str
    columns: list[str]
    rows: list[JsonDict]
    limit: int
    truncated: bool = False


class DBQueryArtifactEnvelope(BackendModel):
    artifact_ref: JsonDict
    preview: JsonDict
    profile: JsonDict
    execution: JsonDict
    warnings: list[str] = []

from __future__ import annotations

from data_agent_backend.models.common import BackendError
from data_agent_backend.models.datasource import DatasourceCredential, DatasourceRecord, DatasourceType
from data_agent_backend.services.connectors.base import DatasourceConnector
from data_agent_backend.services.connectors.mysql_connector import MySQLConnector
from data_agent_backend.services.connectors.postgres_connector import PostgreSQLConnector
from data_agent_backend.services.connectors.sqlite_connector import SQLiteDatasourceConnector


class ConnectorRegistry:
    def create(self, record: DatasourceRecord, credential: DatasourceCredential) -> DatasourceConnector:
        if record.type == DatasourceType.mysql:
            return MySQLConnector(record, credential)
        if record.type == DatasourceType.postgresql:
            return PostgreSQLConnector(record, credential)
        if record.type == DatasourceType.sqlite:
            return SQLiteDatasourceConnector(record, credential)
        raise BackendError("UNSUPPORTED_DATASOURCE_TYPE", "Unsupported datasource type.", {"type": str(record.type)})


default_connector_registry = ConnectorRegistry()

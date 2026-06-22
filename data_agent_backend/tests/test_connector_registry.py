from __future__ import annotations

import pytest

from data_agent_backend.models.common import BackendError
from data_agent_backend.models.datasource import DatasourceCredential, DatasourceRecord, DatasourceType
from data_agent_backend.services.connectors.mysql_connector import MySQLConnector
from data_agent_backend.services.connectors.postgres_connector import PostgreSQLConnector
from data_agent_backend.services.connectors.registry import ConnectorRegistry
from data_agent_backend.services.connectors.sqlite_connector import SQLiteDatasourceConnector


def _record(datasource_type: DatasourceType) -> DatasourceRecord:
    return DatasourceRecord(
        datasource_id="ds_test",
        name="analytics",
        type=datasource_type,
        host="localhost" if datasource_type != DatasourceType.sqlite else None,
        port=3306 if datasource_type == DatasourceType.mysql else 5432 if datasource_type == DatasourceType.postgresql else None,
        database="analytics",
        username="analyst" if datasource_type != DatasourceType.sqlite else None,
        path="/tmp/analytics.sqlite" if datasource_type == DatasourceType.sqlite else None,
        created_at="2026-06-10T00:00:00+00:00",
    )


def test_registry_selects_mysql_connector() -> None:
    connector = ConnectorRegistry().create(_record(DatasourceType.mysql), DatasourceCredential(password="secret"))

    assert isinstance(connector, MySQLConnector)


def test_registry_selects_postgresql_connector() -> None:
    connector = ConnectorRegistry().create(_record(DatasourceType.postgresql), DatasourceCredential(password="secret"))

    assert isinstance(connector, PostgreSQLConnector)


def test_registry_selects_sqlite_datasource_connector() -> None:
    connector = ConnectorRegistry().create(_record(DatasourceType.sqlite), DatasourceCredential())

    assert isinstance(connector, SQLiteDatasourceConnector)


def test_mysql_connector_requires_password_before_dependency_or_connection() -> None:
    connector = ConnectorRegistry().create(_record(DatasourceType.mysql), DatasourceCredential())

    with pytest.raises(BackendError) as exc_info:
        connector.test_connection()

    assert exc_info.value.code == "CREDENTIAL_REQUIRED"

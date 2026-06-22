from __future__ import annotations

import pytest

from data_agent_backend.config import BackendConfig
from data_agent_backend.models.common import BackendError
from data_agent_backend.models.datasource import (
    CatalogSummary,
    ColumnInfo,
    DatasourceCreateRequest,
    DatasourceCredential,
    DatasourceRecord,
    DatasourceType,
    TableSummary,
)
from data_agent_backend.services import factory
from data_agent_backend.services.connectors.mysql_connector import MySQLConnector
from data_agent_backend.services.factory import create_backend_services


MYSQL_ENV_KEYS = (
    "MYSQL_HOST",
    "MYSQL_DATABASE",
    "MYSQL_USERNAME",
    "MYSQL_PASSWORD",
    "MYSQL_PORT",
    "MYSQL_DATASOURCE_NAME",
)


def _services(tmp_path, monkeypatch):
    for key in MYSQL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(factory, "find_dotenv", lambda usecwd=True: "")
    return create_backend_services(BackendConfig(base_data_dir=tmp_path / ".data_agent"))


def _register(services, name: str = "analytics_mysql") -> str:
    record = services.datasource_service.register(
        DatasourceCreateRequest(
            name=name,
            type=DatasourceType.mysql,
            host="localhost",
            port=3306,
            database="analytics",
            username="analyst",
        )
    )
    return record.datasource_id


def _record() -> DatasourceRecord:
    return DatasourceRecord(
        datasource_id="ds_test",
        name="analytics_mysql",
        type=DatasourceType.mysql,
        host="localhost",
        port=3306,
        database="analytics",
        username="analyst",
        created_at="2026-06-10T00:00:00+00:00",
    )


def test_services_isolate_ambient_mysql_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MYSQL_HOST", "ambient-host")
    monkeypatch.setenv("MYSQL_DATABASE", "ambient_database")
    monkeypatch.setenv("MYSQL_USERNAME", "ambient_user")
    monkeypatch.setenv("MYSQL_PASSWORD", "ambient_secret")
    monkeypatch.setenv("MYSQL_PORT", "3307")
    monkeypatch.setenv("MYSQL_DATASOURCE_NAME", "ambient_mysql")

    services = _services(tmp_path, monkeypatch)

    assert services.datasource_service.list_all() == []


def test_services_isolate_dotenv_mysql_config(tmp_path, monkeypatch) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "MYSQL_HOST=dotenv-host",
                "MYSQL_DATABASE=dotenv_database",
                "MYSQL_USERNAME=dotenv_user",
                "MYSQL_PASSWORD=dotenv_secret",
                "MYSQL_PORT=3307",
                "MYSQL_DATASOURCE_NAME=dotenv_mysql",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(factory, "find_dotenv", lambda usecwd=True: str(dotenv_path))

    services = _services(tmp_path, monkeypatch)

    assert services.datasource_service.list_all() == []


def test_resolve_datasource_id_requires_explicit_datasource_id(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path, monkeypatch)

    with pytest.raises(BackendError) as exc_info:
        services.datasource_service.resolve_datasource_id(None)

    assert exc_info.value.code == "DATASOURCE_ID_REQUIRED"


def test_resolve_datasource_id_uses_explicit_registered_datasource(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path, monkeypatch)
    datasource_id = _register(services)

    assert services.datasource_service.resolve_datasource_id(datasource_id) == datasource_id


def test_resolve_datasource_id_rejects_explicit_empty_string(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path, monkeypatch)
    datasource_id = _register(services)

    with pytest.raises(BackendError) as exc_info:
        services.datasource_service.resolve_datasource_id("")

    assert exc_info.value.code == "NOT_FOUND"
    with pytest.raises(BackendError) as missing_info:
        services.datasource_service.resolve_datasource_id(None)
    assert missing_info.value.code == "DATASOURCE_ID_REQUIRED"


def test_resolve_datasource_id_rejects_ambiguous_default(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path, monkeypatch)
    _register(services, "primary")
    services.datasource_service.register(
        DatasourceCreateRequest(
            name="secondary",
            type=DatasourceType.mysql,
            host="localhost",
            port=3307,
            database="analytics_2",
            username="analyst",
        )
    )

    with pytest.raises(BackendError) as exc_info:
        services.datasource_service.resolve_datasource_id(None)

    assert exc_info.value.code == "DATASOURCE_ID_REQUIRED"


def test_list_agent_datasources_hides_connection_secrets(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path, monkeypatch)
    datasource_id = _register(services)

    data = services.datasource_service.list_agent_datasources()
    serialized_data = [item.model_dump(mode="json") for item in data]

    assert serialized_data == [
        {
            "datasource_id": datasource_id,
            "name": "analytics_mysql",
            "type": "mysql",
            "database": "analytics",
            "is_default": False,
        }
    ]
    serialized = str(data)
    assert "secret" not in serialized
    assert "analyst" not in serialized
    assert "localhost" not in serialized


def test_quote_identifier_accepts_simple_mysql_identifier() -> None:
    connector = MySQLConnector(_record(), "secret")

    assert connector.quote_identifier("orders") == "`orders`"


def test_quote_identifier_rejects_unsafe_identifier() -> None:
    connector = MySQLConnector(_record(), "secret")

    with pytest.raises(BackendError) as exc_info:
        connector.quote_identifier("orders; DROP TABLE users")

    assert exc_info.value.code == "INVALID_IDENTIFIER"


class _FakeCatalogCursor:
    def __init__(self, connection) -> None:
        self._connection = connection
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, query: str) -> None:
        self._connection.queries.append(query)
        if query == "SHOW TABLES":
            self._rows = [("orders; DROP TABLE users",)]
            return
        if query.startswith("DESCRIBE"):
            raise AssertionError(f"Unsafe DESCRIBE query was executed: {query}")

    def fetchall(self):
        return self._rows


class _FakeCatalogConnection:
    def __init__(self) -> None:
        self.queries = []
        self.closed = False

    def cursor(self) -> _FakeCatalogCursor:
        return _FakeCatalogCursor(self)

    def close(self) -> None:
        self.closed = True


def test_fetch_catalog_rejects_unsafe_table_name_before_describe(monkeypatch) -> None:
    connector = MySQLConnector(_record(), "secret")
    connection = _FakeCatalogConnection()
    monkeypatch.setattr(connector, "_connect", lambda: connection)

    with pytest.raises(BackendError) as exc_info:
        connector.fetch_catalog()

    assert exc_info.value.code == "INVALID_IDENTIFIER"
    assert connection.queries == ["SHOW TABLES"]
    assert connection.closed is True


class FakeConnector:
    def __init__(self) -> None:
        self.describe_calls: list[str] = []
        self.sample_calls: list[tuple[str, int]] = []

    def fetch_catalog(self) -> CatalogSummary:
        return CatalogSummary(
            datasource_id="ds_test",
            database="analytics",
            tables={
                "orders": TableSummary(
                    columns={
                        "order_id": ColumnInfo(type="BIGINT", nullable=False, key="PRI"),
                        "amount": ColumnInfo(type="DECIMAL(10,2)", nullable=True, key=""),
                    }
                )
            },
            refreshed_at="2026-06-10T00:00:00+00:00",
        )

    def describe_table(self, table_name: str) -> TableSummary:
        self.describe_calls.append(table_name)
        return TableSummary(columns={"order_id": ColumnInfo(type="BIGINT", nullable=False, key="PRI")})

    def sample_rows(self, table_name: str, limit: int):
        self.sample_calls.append((table_name, limit))
        return [(1, "paid")], ["order_id", "status"]


def test_get_schema_returns_agent_schema_summary(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path, monkeypatch)
    datasource_id = _register(services)
    fake = FakeConnector()
    credential = DatasourceCredential(password="secret")
    monkeypatch.setattr(services.datasource_service, "_get_connector", lambda _record, _credential: fake)

    result = services.datasource_service.get_schema(datasource_id, credential)

    assert result.datasource_id == datasource_id
    assert result.database == "analytics"
    assert result.tables[0].name == "orders"
    assert result.tables[0].columns[0].name == "order_id"


def test_describe_table_delegates_to_connector(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path, monkeypatch)
    datasource_id = _register(services)
    fake = FakeConnector()
    credential = DatasourceCredential(password="secret")
    monkeypatch.setattr(services.datasource_service, "_get_connector", lambda _record, _credential: fake)

    result = services.datasource_service.describe_table(datasource_id, "orders", credential)

    assert fake.describe_calls == ["orders"]
    assert result.table_name == "orders"
    assert result.columns[0].name == "order_id"


def test_sample_rows_returns_fixed_preview_shape(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path, monkeypatch)
    datasource_id = _register(services)
    fake = FakeConnector()
    credential = DatasourceCredential(password="secret")
    monkeypatch.setattr(services.datasource_service, "_get_connector", lambda _record, _credential: fake)

    result = services.datasource_service.sample_rows(datasource_id, "orders", 10, credential)

    assert fake.sample_calls == [("orders", 10)]
    assert result.columns == ["order_id", "status"]
    assert result.rows == [{"order_id": 1, "status": "paid"}]
    assert result.limit == 10
    assert result.truncated is False


def test_sample_rows_rejects_explicit_zero_limit(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path, monkeypatch)
    datasource_id = _register(services)
    fake = FakeConnector()
    monkeypatch.setattr(services.datasource_service, "_get_connector", lambda _record, _credential: fake)

    with pytest.raises(BackendError) as exc_info:
        services.datasource_service.sample_rows(datasource_id, "orders", 0, DatasourceCredential(password="secret"))

    assert exc_info.value.code == "POLICY_BLOCKED"
    assert fake.sample_calls == []


def test_query_datasource_redacts_connector_exception_details(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path, monkeypatch)
    datasource_id = _register(services)

    class FailingConnector:
        def execute_query(self, query: str, row_limit: int):
            raise RuntimeError("Access denied for user analyst at localhost with password secret")

    monkeypatch.setattr(services.datasource_service, "_get_connector", lambda _record, _credential: FailingConnector())

    with pytest.raises(BackendError) as exc_info:
        services.datasource_service.query_datasource(datasource_id, DatasourceCredential(password="secret"), "SELECT 1", 10)

    assert exc_info.value.code == "DATASOURCE_QUERY_FAILED"
    serialized_details = str(exc_info.value.details)
    assert "analyst" not in serialized_details
    assert "localhost" not in serialized_details
    assert "secret" not in serialized_details
    assert "Access denied" not in serialized_details

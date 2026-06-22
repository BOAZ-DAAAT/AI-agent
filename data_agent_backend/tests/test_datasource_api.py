from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient

from data_agent_backend.api.app import create_app
from data_agent_backend.config import BackendConfig
from data_agent_backend.services.factory import create_backend_services


def _services(tmp_path):
    return create_backend_services(BackendConfig(base_data_dir=tmp_path / ".data_agent"))


def _sqlite_db(tmp_path):
    path = tmp_path / "external.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE orders(order_id INTEGER PRIMARY KEY, status TEXT, amount REAL)")
        conn.executemany(
            "INSERT INTO orders(order_id, status, amount) VALUES (?, ?, ?)",
            [(1, "paid", 10.5), (2, "pending", 7.25)],
        )
    return path


def test_datasource_create_rejects_password_field(tmp_path) -> None:
    client = TestClient(create_app(_services(tmp_path)))

    response = client.post(
        "/datasources",
        json={
            "name": "analytics_mysql",
            "type": "mysql",
            "host": "localhost",
            "port": 3306,
            "database": "analytics",
            "username": "analyst",
            "password": "secret",
        },
    )

    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"]["code"] == "VALIDATION_ERROR"


def test_datasource_api_hides_host_username_and_password(tmp_path) -> None:
    client = TestClient(create_app(_services(tmp_path)))

    created = client.post(
        "/datasources",
        json={
            "name": "analytics_mysql",
            "type": "mysql",
            "host": "localhost",
            "port": 3306,
            "database": "analytics",
            "username": "analyst",
        },
    ).json()
    listed = client.get("/datasources").json()

    assert created["ok"] is True
    assert listed["ok"] is True
    serialized = str(listed["data"])
    assert "localhost" not in serialized
    assert "analyst" not in serialized
    assert "secret" not in serialized


def test_datasource_schema_requires_credential_payload(tmp_path) -> None:
    client = TestClient(create_app(_services(tmp_path)))
    datasource_id = client.post(
        "/datasources",
        json={
            "name": "analytics_sqlite",
            "type": "sqlite",
            "path": str(_sqlite_db(tmp_path)),
        },
    ).json()["data"]["datasource_id"]

    response = client.post(f"/datasources/{datasource_id}/schema", json={})

    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"]["code"] == "VALIDATION_ERROR"


def test_sqlite_datasource_api_schema_sample_and_sql_execution(tmp_path) -> None:
    services = _services(tmp_path)
    client = TestClient(create_app(services))
    db_path = _sqlite_db(tmp_path)
    datasource_id = client.post(
        "/datasources",
        json={"name": "analytics_sqlite", "type": "sqlite", "path": str(db_path)},
    ).json()["data"]["datasource_id"]
    run = services.run_service.create_run()

    schema = client.post(f"/datasources/{datasource_id}/schema", json={"credential": {}}).json()
    sample = client.post(f"/datasources/{datasource_id}/tables/orders/sample", json={"credential": {}, "limit": 1}).json()
    result = client.post(
        "/execution/sql",
        json={
            "run_id": run.run_id,
            "datasource_id": datasource_id,
            "credential": {},
            "query": "SELECT order_id, status FROM orders ORDER BY order_id",
            "row_limit": 1,
        },
    ).json()

    assert schema["ok"] is True
    assert schema["data"]["tables"][0]["name"] == "orders"
    assert sample["ok"] is True
    assert sample["data"]["rows"] == [{"order_id": 1, "status": "paid", "amount": 10.5}]
    assert result["ok"] is True
    artifact = services.artifact_registry.get_artifact(result["data"]["artifact_id"])
    assert artifact.metadata["datasource_id"] == datasource_id
    assert artifact.metadata["row_limit"] == 1
    assert artifact.metadata["returned_rows"] == 1
    assert "credential" not in str(artifact.metadata)

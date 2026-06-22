from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from data_agent_backend.api.app import create_app
from data_agent_backend.config import BackendConfig
from data_agent_backend.models.common import BackendError
from data_agent_backend.models.contexts import PolicyContext
from data_agent_backend.models.datasource import DatasourceCreateRequest, DatasourceCredential, DatasourceType
from data_agent_backend.models.policy import PolicyDecision, RiskLevel
from data_agent_backend.services.factory import create_backend_services


def _create_services(tmp_path):
    return create_backend_services(BackendConfig(base_data_dir=tmp_path / ".data_agent"))


def _register_datasource(services) -> str:
    record = services.datasource_service.register(
        DatasourceCreateRequest(
            name="analytics_mysql",
            type=DatasourceType.mysql,
            host="localhost",
            port=3306,
            database="analytics",
            username="analyst",
        )
    )
    return record.datasource_id


def test_run_sql_query_requires_datasource_id(tmp_path) -> None:
    services = _create_services(tmp_path)
    run = services.run_service.create_run()

    with pytest.raises(BackendError) as exc_info:
        services.sql_executor.run_sql_query("SELECT 1 AS value", run.run_id)

    assert exc_info.value.code == "DATASOURCE_ID_REQUIRED"


def test_run_sql_query_uses_datasource_service_and_records_metadata(tmp_path, monkeypatch) -> None:
    services = _create_services(tmp_path)
    datasource_id = _register_datasource(services)
    run = services.run_service.create_run()
    calls = []

    credential = DatasourceCredential(password="secret")

    def fake_query_datasource(received_datasource_id: str, received_credential, query: str, row_limit: int):
        calls.append((received_datasource_id, received_credential, query, row_limit))
        return [(42,)], ["answer"], "answer\r\n42\r\n"

    monkeypatch.setattr(services.datasource_service, "query_datasource", fake_query_datasource)

    ref = services.sql_executor.run_sql_query(
        "SELECT 42 AS answer",
        run.run_id,
        datasource_id=datasource_id,
        credential=credential,
        row_limit=25,
    )

    assert calls == [(datasource_id, credential, "SELECT 42 AS answer", 25)]
    assert not hasattr(services.sql_executor, "_execute_duckdb")
    assert not hasattr(services.sql_executor, "_execute_sqlite")

    artifact = services.artifact_registry.get_artifact(ref.artifact_id)
    assert artifact.metadata["datasource_id"] == datasource_id
    assert artifact.metadata["row_limit"] == 25
    assert artifact.metadata["returned_rows"] == 1
    assert artifact.preview["columns"] == ["answer"]


def test_execution_sql_api_rejects_missing_datasource_id(tmp_path) -> None:
    services = _create_services(tmp_path)
    run = services.run_service.create_run()
    client = TestClient(create_app(services))

    response = client.post("/execution/sql", json={"query": "SELECT 1 AS value", "run_id": run.run_id})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"]["code"] == "VALIDATION_ERROR"


def test_execution_sql_api_rejects_connection_id_field(tmp_path) -> None:
    services = _create_services(tmp_path)
    run = services.run_service.create_run()
    client = TestClient(create_app(services))

    response = client.post(
        "/execution/sql",
        json={"query": "SELECT 1 AS value", "run_id": run.run_id, "connection_id": "ds_legacy"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"]["code"] == "VALIDATION_ERROR"


def test_run_analysis_query_returns_artifact_envelope(tmp_path, monkeypatch) -> None:
    services = _create_services(tmp_path)
    datasource_id = _register_datasource(services)
    run = services.run_service.create_run()

    credential = DatasourceCredential(password="secret")

    def fake_query_datasource(received_datasource_id: str, received_credential, query: str, row_limit: int):
        assert received_datasource_id == datasource_id
        assert received_credential == credential
        assert query == "SELECT 42 AS answer"
        assert row_limit == 25
        return [(42,)], ["answer"], "answer\r\n42\r\n"

    monkeypatch.setattr(services.datasource_service, "query_datasource", fake_query_datasource)

    envelope = services.sql_executor.run_analysis_query(
        query="SELECT 42 AS answer",
        run_id=run.run_id,
        datasource_id=datasource_id,
        credential=credential,
        row_limit=25,
    )

    assert envelope["artifact_ref"]["type"] == "sql_result"
    assert envelope["preview"]["columns"] == ["answer"]
    assert envelope["preview"]["rows"] == [{"answer": 42}]
    assert envelope["profile"]["returned_rows"] == 1
    assert envelope["execution"]["datasource_id"] == datasource_id
    assert envelope["execution"]["tool_name"] == "db_run_analysis_query"
    assert envelope["warnings"] == []


def test_run_analysis_query_rejects_blocked_sql(tmp_path) -> None:
    services = _create_services(tmp_path)
    datasource_id = _register_datasource(services)
    run = services.run_service.create_run()

    with pytest.raises(BackendError) as exc_info:
        services.sql_executor.run_analysis_query(
            query="DROP TABLE users",
            run_id=run.run_id,
            datasource_id=datasource_id,
            credential=DatasourceCredential(password="secret"),
        )

    assert exc_info.value.code == "POLICY_BLOCKED"


def test_run_analysis_query_rejects_explicit_zero_row_limit_before_execution(tmp_path, monkeypatch) -> None:
    services = _create_services(tmp_path)
    datasource_id = _register_datasource(services)
    run = services.run_service.create_run()

    def fail_query_datasource(received_datasource_id: str, received_credential, query: str, row_limit: int):
        raise AssertionError("query_datasource should not be called for invalid row_limit")

    monkeypatch.setattr(services.datasource_service, "query_datasource", fail_query_datasource)

    with pytest.raises(BackendError) as exc_info:
        services.sql_executor.run_analysis_query(
            "SELECT 1 AS value",
            run.run_id,
            datasource_id=datasource_id,
            credential=DatasourceCredential(password="secret"),
            row_limit=0,
        )

    assert exc_info.value.code == "POLICY_BLOCKED"


def test_run_analysis_query_raises_approval_required_before_policy_blocked(tmp_path, monkeypatch) -> None:
    services = _create_services(tmp_path)
    services.config.max_sql_row_limit_without_approval = 10
    datasource_id = _register_datasource(services)
    run = services.run_service.create_run()

    def fail_query_datasource(received_datasource_id: str, received_credential, query: str, row_limit: int):
        raise AssertionError("query_datasource should not be called when approval is required")

    monkeypatch.setattr(services.datasource_service, "query_datasource", fail_query_datasource)

    with pytest.raises(BackendError) as exc_info:
        services.sql_executor.run_analysis_query(
            query="SELECT 1 AS value",
            run_id=run.run_id,
            datasource_id=datasource_id,
            credential=DatasourceCredential(password="secret"),
            row_limit=11,
        )

    assert exc_info.value.code == "APPROVAL_REQUIRED"


def test_run_analysis_query_forces_policy_context_run_id_and_tool_name(tmp_path, monkeypatch) -> None:
    services = _create_services(tmp_path)
    datasource_id = _register_datasource(services)
    stale_run = services.run_service.create_run()
    actual_run = services.run_service.create_run()
    captured_contexts = []

    def fake_evaluate(action, resource="", payload=None, context=None):
        captured_contexts.append(context)
        return PolicyDecision(
            decision_id="pd_allow",
            allowed=True,
            requires_approval=False,
            risk_level=RiskLevel.low,
            reason="allowed",
        )

    def fake_query_datasource(received_datasource_id: str, received_credential, query: str, row_limit: int):
        return [(1,)], ["value"], "value\r\n1\r\n"

    monkeypatch.setattr(services.policy_engine, "evaluate", fake_evaluate)
    monkeypatch.setattr(services.datasource_service, "query_datasource", fake_query_datasource)

    services.sql_executor.run_analysis_query(
        query="SELECT 1 AS value",
        run_id=actual_run.run_id,
        datasource_id=datasource_id,
        credential=DatasourceCredential(password="secret"),
        context=PolicyContext(run_id=stale_run.run_id, tool_name="some_other_tool"),
    )

    assert captured_contexts
    assert captured_contexts[0].run_id == actual_run.run_id
    assert captured_contexts[0].tool_name == "db_run_analysis_query"

from __future__ import annotations

from fastapi.testclient import TestClient

from data_agent_backend.api.app import create_app
from data_agent_backend.config import BackendConfig
from data_agent_backend.services.factory import create_backend_services, create_core_services


def _services(tmp_path):
    return create_backend_services(BackendConfig(base_data_dir=tmp_path / ".data_agent"))


def test_create_core_services_excludes_optional_services(tmp_path) -> None:
    services = create_core_services(BackendConfig(base_data_dir=tmp_path / ".data_agent"))

    assert hasattr(services, "datasource_service")
    assert hasattr(services, "sql_executor")
    assert hasattr(services, "sandbox_executor")
    assert hasattr(services, "artifact_registry")
    assert hasattr(services, "run_service")
    assert hasattr(services, "policy_engine")
    assert not hasattr(services, "memory_store")
    assert not hasattr(services, "approval_store")
    assert not hasattr(services, "workspace_backend")
    assert not hasattr(services, "export_service")
    assert not hasattr(services, "catalog_store")
    assert not hasattr(services, "checkpoint_manager")


def test_runs_api_uses_tool_result_envelope(tmp_path) -> None:
    client = TestClient(create_app(_services(tmp_path)))

    created = client.post("/runs/create", json={"metadata": {"source": "api-test"}}).json()
    listed = client.post("/runs/list", json={}).json()

    assert created["ok"] is True
    assert created["data"]["run_id"].startswith("run_")
    assert listed["ok"] is True
    assert [item["run_id"] for item in listed["data"]] == [created["data"]["run_id"]]


def test_execution_python_api_calls_sandbox_service_without_mcp(tmp_path) -> None:
    services = _services(tmp_path)
    client = TestClient(create_app(services))
    run_id = services.run_service.create_run().run_id

    result = client.post(
        "/execution/python",
        json={"run_id": run_id, "code": "print('hello')", "context": {"approval_id": "appr_test"}},
    ).json()

    assert result["ok"] is True
    assert result["data"]["status"] == "sandbox_not_configured"
    assert result["data"]["error_message"] == "Python sandbox execution is disabled in this MVP."


def test_default_app_excludes_optional_routes(tmp_path) -> None:
    client = TestClient(create_app(_services(tmp_path)))

    optional_paths = [
        "/workspace/read-text",
        "/memory/list",
        "/approvals/pending",
        "/exports/create",
        "/catalog/list",
    ]
    for path in optional_paths:
        assert client.post(path, json={}).status_code == 404

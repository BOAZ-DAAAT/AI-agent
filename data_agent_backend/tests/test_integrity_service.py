from __future__ import annotations

import sys
import pytest
import types

TestClient = pytest.importorskip("fastapi.testclient").TestClient

from data_agent_backend.api.app import create_app
from data_agent_backend.config import BackendConfig
from data_agent_backend.models.common import BackendError
from data_agent_backend.models.integrity import IntegrityCheckResult, IntegrityStatus
from data_agent_backend.services.integrity_service import DefaultIntegrityRunner
from data_agent_backend.services.factory import create_backend_services


class FakeIntegrityRunner:
    def __init__(self, statuses: dict[str, IntegrityStatus]) -> None:
        self.statuses = statuses
        self.calls: list[tuple[str, list[str] | None, str | None, dict]] = []

    def run_checks(self, dataset_name, tables, source_version, metadata=None):
        self.calls.append((dataset_name, tables, source_version, metadata or {}))
        return [
            IntegrityCheckResult(
                table_name=table,
                status=self.statuses.get(table, IntegrityStatus.pass_),
                severity="error" if self.statuses.get(table) == IntegrityStatus.fail else "info",
                summary={"message": f"{table} checked"},
                artifact_refs=[{"uri": f"artifact://{table}"}] if self.statuses.get(table) == IntegrityStatus.fail else [],
            )
            for table in (tables or [])
        ]


def _services(tmp_path):
    return create_backend_services(BackendConfig(base_data_dir=tmp_path / ".data_agent"))


def test_notify_persists_queue_and_compact_summary_survives_restart(tmp_path) -> None:
    services = _services(tmp_path)
    notify = services.integrity_service.notify_dataset_update("warehouse", tables={"orders": 2, "customers": 1}, source_version="v1")

    assert notify.ok is True
    assert notify.data["status"] == "queued"
    initial_summary = services.integrity_service.get_summary("warehouse")
    assert initial_summary.ok is True
    assert initial_summary.data["status"] == "stale"
    assert initial_summary.data["attention_counts"]["stale"] == 2
    assert initial_summary.data["pending_jobs"][0]["job_id"] == notify.data["job_id"]

    restarted = _services(tmp_path)
    restarted_summary = restarted.integrity_service.get_summary("warehouse")
    assert restarted_summary.ok is True
    assert restarted_summary.data["pending_jobs"][0]["job_id"] == notify.data["job_id"]


def test_ensure_tables_ready_runs_priority_scoped_check_with_fake_runner(tmp_path) -> None:
    services = _services(tmp_path)
    fake = FakeIntegrityRunner({"orders": IntegrityStatus.pass_, "customers": IntegrityStatus.fail})
    services.integrity_service.runner = fake

    ready = services.integrity_service.ensure_tables_ready("warehouse", ["orders", "customers"], wait_timeout_s=1.0)

    assert ready.ok is True
    assert ready.data["ready"] is False
    assert ready.data["status_by_table"]["orders"]["status"] == "pass"
    assert ready.data["status_by_table"]["customers"]["status"] == "fail"
    assert fake.calls[0][1] == ["orders", "customers"]
    summary = services.integrity_service.get_summary("warehouse")
    assert summary.ok is True
    assert [item["table_name"] for item in summary.data["summaries"]] == ["customers"]
    assert summary.data["summaries"][0]["artifact_refs"] == [{"uri": "artifact://customers"}]


def test_integrity_api_uses_tool_result_envelope_and_excludes_pass_by_default(tmp_path) -> None:
    services = _services(tmp_path)
    services.integrity_service.runner = FakeIntegrityRunner({"orders": IntegrityStatus.pass_})
    client = TestClient(create_app(services))

    notify = client.post(
        "/integrity/notify-dataset-update",
        json={"dataset_name": "warehouse", "tables": {"orders": 3}, "source_version": "v1"},
    ).json()
    processed = client.post("/integrity/process-next").json()
    summary = client.post("/integrity/summary", json={"dataset_name": "warehouse"}).json()
    full_summary = client.post("/integrity/summary", json={"dataset_name": "warehouse", "include_pass": True}).json()

    assert notify["ok"] is True
    assert processed["ok"] is True
    assert processed["data"]["processed"] is True
    assert summary["ok"] is True
    assert summary["data"]["summaries"] == []
    assert full_summary["data"]["summaries"][0]["status"] == "pass"


def test_default_runner_uses_real_integrity_runner_when_available(monkeypatch) -> None:
    fake_module = types.ModuleType("DATA_Analyst_Assistant_Agent.agents.sql.db.integrity_runner")

    def fake_run_dataset_integrity_checks(*, dataset_name, table_scope, source_version, metadata):
        assert dataset_name == "warehouse"
        assert table_scope == ["orders"]
        return {
            "tables": {
                "orders": {
                    "status": "fail",
                    "physical_status": "fail",
                    "semantic_status": "warning",
                    "failures": [{"status": "FAIL", "intent": "pk broken"}],
                    "temporary_fix_artifacts": [{"kind": "sql_patch"}],
                    "root_cause_summary": ["duplicate primary key"],
                }
            }
        }

    fake_module.run_dataset_integrity_checks = fake_run_dataset_integrity_checks
    monkeypatch.setitem(sys.modules, "DATA_Analyst_Assistant_Agent.agents.sql.db.integrity_runner", fake_module)

    results = DefaultIntegrityRunner().run_checks("warehouse", ["orders"], "v1", {})

    assert len(results) == 1
    assert results[0].table_name == "orders"
    assert results[0].status == IntegrityStatus.fail
    assert results[0].summary["physical_status"] == "fail"
    assert results[0].summary["root_cause_summary"] == ["duplicate primary key"]


def test_default_runner_raises_backend_error_when_real_runner_raises(monkeypatch) -> None:
    fake_module = types.ModuleType("DATA_Analyst_Assistant_Agent.agents.sql.db.integrity_runner")

    def fake_run_dataset_integrity_checks(*, dataset_name, table_scope, source_version, metadata):
        raise RuntimeError("boom")

    fake_module.run_dataset_integrity_checks = fake_run_dataset_integrity_checks
    monkeypatch.setitem(sys.modules, "DATA_Analyst_Assistant_Agent.agents.sql.db.integrity_runner", fake_module)

    with pytest.raises(BackendError) as exc_info:
        DefaultIntegrityRunner().run_checks("warehouse", ["orders"], "v1", {})

    assert exc_info.value.code == "INTEGRITY_RUN_FAILED"
    assert "runner_error" in exc_info.value.details


def test_ensure_tables_ready_enqueues_when_unrelated_job_is_pending(tmp_path) -> None:
    services = _services(tmp_path)
    fake = FakeIntegrityRunner({"orders": IntegrityStatus.pass_, "customers": IntegrityStatus.pass_})
    services.integrity_service.runner = fake

    notify = services.integrity_service.notify_dataset_update("warehouse", tables={"customers": 1}, source_version="v1")
    assert notify.ok is True

    ready = services.integrity_service.ensure_tables_ready("warehouse", ["orders"], wait_timeout_s=1.0)

    assert ready.ok is True
    assert fake.calls[-1][1] == ["orders"]
    summary = services.integrity_service.get_summary("warehouse", tables=["orders"], include_pass=True)
    assert summary.ok is True
    assert summary.data["summaries"][0]["table_name"] == "orders"


def test_scoped_ensure_tables_ready_inserts_pending_summary_without_overwriting_ready_version(tmp_path) -> None:
    services = _services(tmp_path)
    fake = FakeIntegrityRunner({"orders": IntegrityStatus.pass_})
    services.integrity_service.runner = fake

    ready = services.integrity_service.ensure_tables_ready("warehouse", ["orders"], wait_timeout_s=0.0)
    assert ready.ok is True
    assert ready.data["ready"] is False

    summary = services.integrity_service.get_summary("warehouse", tables=["orders"], include_pass=True)
    assert summary.ok is True
    assert summary.data["summaries"][0]["table_name"] == "orders"
    assert summary.data["summaries"][0]["status"] == "pending"


def test_failed_integrity_result_marks_job_failed(tmp_path) -> None:
    services = _services(tmp_path)
    fake = FakeIntegrityRunner({"orders": IntegrityStatus.fail})
    services.integrity_service.runner = fake

    result = services.integrity_service.ensure_tables_ready("warehouse", ["orders"], wait_timeout_s=1.0)

    assert result.ok is True
    assert result.data["ready"] is False
    jobs = services.integrity_service.get_summary("warehouse", tables=["orders"], include_pass=True)
    assert jobs.ok is True
    assert jobs.data["pending_jobs"] == []
    row = services.sqlite.query_one("SELECT status, error_json FROM integrity_queue ORDER BY created_at DESC LIMIT 1")
    assert row["status"] == "failed"
    assert row["error_json"] is not None

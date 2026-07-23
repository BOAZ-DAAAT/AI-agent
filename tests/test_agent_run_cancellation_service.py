from __future__ import annotations

from types import SimpleNamespace

from backend.agent_runs.service import cancel_agent_run, launch_agent_run
from data_agent_backend.config import BackendConfig
from data_agent_backend.services.factory import create_backend_services


def _services(tmp_path):
    return create_backend_services(BackendConfig(base_data_dir=tmp_path / ".data_agent"))


def test_cancelled_created_run_never_launches(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path)
    run = services.run_service.create_run(
        thread_id="thread_cancelled_before_launch",
        project_id="sess_001",
        metadata={"session_id": "sess_001"},
    )
    cancel_agent_run(services=services, run_id=run.run_id, cancelled_by="dev")
    catalog_called = False

    def fail_if_catalog_is_built(session):
        nonlocal catalog_called
        catalog_called = True
        raise AssertionError("cancelled run must not initialize the workflow")

    monkeypatch.setattr(
        "backend.agent_runs.service.build_session_catalog_summary",
        fail_if_catalog_is_built,
    )

    launch_agent_run(
        services=services,
        session=SimpleNamespace(id="sess_001", session_db="session_db"),
        username="dev",
        query="분석해줘",
        run_id=run.run_id,
        thread_id="thread_cancelled_before_launch",
    )

    assert catalog_called is False
    assert services.run_service.get_run(run.run_id).status.value == "cancelled"
    event_types = [event.event_type for event in services.run_service.list_events(run.run_id)]
    assert event_types == ["run.cancelled"]


def test_cancelled_run_is_not_marked_failed_by_late_worker_error(tmp_path) -> None:
    from backend.agent_runs.service import _mark_run_failed

    services = _services(tmp_path)
    run = services.run_service.create_run(
        thread_id="thread_late_error",
        project_id="sess_001",
    )
    services.run_service.update_status(run.run_id, "running")
    cancel_agent_run(services=services, run_id=run.run_id, cancelled_by="dev")

    _mark_run_failed(services, run.run_id, "sess_001", RuntimeError("late error"))

    assert services.run_service.get_run(run.run_id).status.value == "cancelled"
    assert not any(
        event.event_type == "run.failed"
        for event in services.run_service.list_events(run.run_id)
    )

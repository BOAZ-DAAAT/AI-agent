from __future__ import annotations

from data_agent_backend.config import BackendConfig
from data_agent_backend.services.factory import create_backend_services


def _run_service(tmp_path):
    services = create_backend_services(
        BackendConfig(base_data_dir=tmp_path / ".data_agent")
    )
    run = services.run_service.create_run(thread_id="thread_event_001")
    return services.run_service, run


def test_append_event_reuses_existing_event_for_same_event_key(tmp_path) -> None:
    run_service, run = _run_service(tmp_path)

    first = run_service.append_event(
        run.run_id,
        "agent.started",
        "SQL Agent started",
        event_key="agent-node:node_001:agent.started:attempt:1",
        node_name="sql_agent",
    )
    duplicate = run_service.append_event(
        run.run_id,
        "agent.started",
        "duplicate delivery",
        event_key="agent-node:node_001:agent.started:attempt:1",
        node_name="sql_agent",
    )

    assert duplicate.event_id == first.event_id
    assert duplicate.message == "SQL Agent started"
    assert duplicate.event_key == first.event_key
    assert len(run_service.list_events(run.run_id)) == 1


def test_append_event_without_event_key_remains_append_only(tmp_path) -> None:
    run_service, run = _run_service(tmp_path)

    first = run_service.append_event(run.run_id, "run.progress", "progress")
    second = run_service.append_event(run.run_id, "run.progress", "progress")

    assert first.event_id != second.event_id
    assert first.event_key is None
    assert second.event_key is None
    assert len(run_service.list_events(run.run_id)) == 2

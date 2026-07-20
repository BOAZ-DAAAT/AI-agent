from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

TestClient = pytest.importorskip("fastapi.testclient").TestClient

from backend.main import create_app
from data_agent_backend.config import BackendConfig
from data_agent_backend.services.factory import create_backend_services


def _services(tmp_path):
    return create_backend_services(BackendConfig(base_data_dir=tmp_path / ".data_agent"))


def _user_header() -> dict[str, str]:
    return {"Authorization": "Bearer ignored"}


def test_create_agent_run_returns_run_id_and_uses_selected_session(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path)
    app = create_app(services=services)
    client = TestClient(app)

    monkeypatch.setattr("backend.auth.deps.Auth.ENABLED", False)
    monkeypatch.setattr(
        "backend.agent_runs.routes.get_owned_session",
        lambda session_id, username: SimpleNamespace(id=session_id, session_db="session_db", mart_db="mart_db"),
    )

    seen: dict[str, str] = {}

    def fake_launch(*, services, session, username, query, run_id, thread_id) -> None:
        seen["session_id"] = session.id
        seen["username"] = username
        seen["query"] = query
        seen["run_id"] = run_id
        seen["thread_id"] = thread_id
        services.run_service.update_status(run_id, "running", metadata={"started": True})
        services.run_service.append_event(run_id, "run.started", "workflow started", node_name="supervisor")

    monkeypatch.setattr("backend.agent_runs.routes.launch_agent_run", fake_launch)

    response = client.post(
        "/agent-runs",
        json={"session_id": "sess_001", "query": "지난 3개월 매출 추이"},
        headers=_user_header(),
    )

    assert response.status_code == 202
    body = response.json()
    assert body["run_id"].startswith("run_")
    assert body["thread_id"].startswith("thread_")
    assert body["status"] == "created"
    assert seen["session_id"] == "sess_001"
    assert seen["username"] == "dev"
    assert seen["query"] == "지난 3개월 매출 추이"

    run_response = client.get(f"/agent-runs/{body['run_id']}", headers=_user_header())
    assert run_response.status_code == 200
    assert run_response.json()["status"] == "running"


def test_get_agent_run_and_events_returns_plain_json(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path)
    app = create_app(services=services)
    client = TestClient(app)

    monkeypatch.setattr("backend.auth.deps.Auth.ENABLED", False)

    run = services.run_service.create_run(thread_id="thread_demo", metadata={"query": "demo"})
    services.run_service.update_status(run.run_id, "running")
    services.run_service.append_event(run.run_id, "run.started", "workflow started", node_name="supervisor")

    run_response = client.get(f"/agent-runs/{run.run_id}", headers=_user_header())
    events_response = client.get(f"/agent-runs/{run.run_id}/events", headers=_user_header())

    assert run_response.status_code == 200
    assert run_response.json()["run_id"] == run.run_id
    assert run_response.json()["status"] == "running"

    assert events_response.status_code == 200
    assert events_response.json()[0]["event_type"] == "run.started"
    assert events_response.json()[0]["node_name"] == "supervisor"


def test_stream_agent_run_events_resumes_after_last_event_id(
    tmp_path,
    monkeypatch,
) -> None:
    services = _services(tmp_path)
    app = create_app(services=services)
    client = TestClient(app)

    monkeypatch.setattr("backend.auth.deps.Auth.ENABLED", False)
    monkeypatch.setattr(
        "backend.agent_runs.routes.EVENT_STREAM_POLL_INTERVAL_SECONDS",
        0,
    )

    run = services.run_service.create_run(thread_id="thread_stream")
    first = services.run_service.append_event(
        run.run_id,
        "agent.started",
        "SQL Agent started",
        node_name="sql_agent",
    )
    second = services.run_service.append_event(
        run.run_id,
        "agent.completed",
        "SQL Agent completed",
        node_name="sql_agent",
    )
    services.run_service.update_status(run.run_id, "succeeded")

    response = client.get(
        f"/agent-runs/{run.run_id}/events/stream",
        headers={**_user_header(), "Last-Event-ID": first.event_id},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert f"id: {first.event_id}" not in response.text
    assert f"id: {second.event_id}" in response.text
    assert "event: run.event" in response.text
    assert "event: run.closed" in response.text

    data_lines = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert data_lines[0]["event_id"] == second.event_id
    assert data_lines[0]["event_type"] == "agent.completed"
    assert data_lines[-1] == {"run_id": run.run_id, "status": "succeeded"}


def test_stream_agent_run_events_replays_all_for_unknown_cursor(
    tmp_path,
    monkeypatch,
) -> None:
    services = _services(tmp_path)
    app = create_app(services=services)
    client = TestClient(app)

    monkeypatch.setattr("backend.auth.deps.Auth.ENABLED", False)
    monkeypatch.setattr(
        "backend.agent_runs.routes.EVENT_STREAM_POLL_INTERVAL_SECONDS",
        0,
    )

    run = services.run_service.create_run(thread_id="thread_replay")
    event = services.run_service.append_event(
        run.run_id,
        "agent.started",
        "SQL Agent started",
        node_name="sql_agent",
    )
    services.run_service.update_status(run.run_id, "failed")

    response = client.get(
        f"/agent-runs/{run.run_id}/events/stream?after=unknown_event",
        headers=_user_header(),
    )

    assert response.status_code == 200
    assert f"id: {event.event_id}" in response.text
    assert "event: run.closed" in response.text

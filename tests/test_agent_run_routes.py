from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

TestClient = pytest.importorskip("fastapi.testclient").TestClient

from backend.main import create_app
from data_agent_backend.config import BackendConfig
from data_agent_backend.models.artifacts import ArtifactRegisterRequest, ArtifactType
from data_agent_backend.services.factory import create_backend_services


def _services(tmp_path):
    return create_backend_services(BackendConfig(base_data_dir=tmp_path / ".data_agent"))


def _user_header() -> dict[str, str]:
    return {"Authorization": "Bearer ignored"}


def _node_summary_payload() -> dict:
    return {
        "title": "월별 매출 SQL 결과",
        "subtitle": "주문 결제 데이터를 월 단위로 집계했습니다.",
        "background": "월별 매출 추이를 확인하기 위한 데이터 마트를 구성했습니다.",
        "code_used": "SELECT month, SUM(payment_value) FROM payments GROUP BY month",
        "detail": {
            "kind": "sql",
            "source_tables": ["order_payments"],
            "integrity_checks": ["결제 금액 결측치 확인"],
            "derived_columns": [],
            "mart_grain": "월",
            "mart_columns": ["month", "revenue"],
            "mart_preview": [],
            "sql_snippet": "SELECT month, SUM(payment_value) FROM payments GROUP BY month",
        },
        "conclusion": "월별 매출 분석에 사용할 집계 결과가 준비되었습니다.",
        "key_finding": "월별 매출 집계가 완료되었습니다.",
        "source_kind": "sql_result",
        "fallback_used": False,
    }


def _register_file_artifact(services, run_id: str, *, payload: dict, metadata: dict):
    return services.artifact_registry.register_artifact(
        ArtifactRegisterRequest(
            run_id=run_id,
            type=ArtifactType.file,
            content_text=json.dumps(payload, ensure_ascii=False),
            filename="artifact.json",
            created_by_tool="test",
            metadata=metadata,
        )
    )


def _waiting_clarification_run(services, *, interrupt_type: str = "clarification"):
    run = services.run_service.create_run(
        thread_id="thread_clarification",
        project_id="sess_001",
        metadata={"session_id": "sess_001"},
    )
    services.run_service.update_status(run.run_id, "running")
    return services.run_service.update_status(
        run.run_id,
        "waiting_input",
        metadata={"interrupt_type": interrupt_type, "node": "collect_clarification"},
    )


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


def test_get_completed_node_summary_returns_registered_summary(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path)
    app = create_app(services=services)
    client = TestClient(app)

    monkeypatch.setattr("backend.auth.deps.Auth.ENABLED", False)
    monkeypatch.setattr(
        "backend.agent_runs.routes.get_owned_session",
        lambda session_id, username: SimpleNamespace(id=session_id),
    )
    run = services.run_service.create_run(thread_id="thread_summary", project_id="sess_001")
    summary_artifact = _register_file_artifact(
        services,
        run.run_id,
        payload=_node_summary_payload(),
        metadata={"kind": "node_summary", "source_artifact_ids": ["art_source"]},
    )
    services.run_service.append_event(
        run.run_id,
        "agent.completed",
        "SQL Agent completed",
        node_name="sql_agent",
        artifact_ids=["art_source"],
        metadata={
            "node_id": "node_001",
            "summary": {
                "agent": "sql_agent",
                "artifact_ids": ["art_source"],
                "summary_artifact_id": summary_artifact.artifact_id,
            },
        },
    )

    response = client.get(
        f"/agent-runs/{run.run_id}/nodes/node_001/summary",
        headers=_user_header(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["node_id"] == "node_001"
    assert body["agent_name"] == "sql_agent"
    assert body["summary_artifact_id"] == summary_artifact.artifact_id
    assert body["summary"]["detail"]["kind"] == "sql"
    assert body["summary"]["key_finding"] == "월별 매출 집계가 완료되었습니다."


def test_get_completed_node_summary_resolves_legacy_event_by_source_artifacts(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path)
    app = create_app(services=services)
    client = TestClient(app)

    monkeypatch.setattr("backend.auth.deps.Auth.ENABLED", False)
    monkeypatch.setattr(
        "backend.agent_runs.routes.get_owned_session",
        lambda session_id, username: SimpleNamespace(id=session_id),
    )
    run = services.run_service.create_run(thread_id="thread_legacy", project_id="sess_001")
    source_artifact = _register_file_artifact(
        services,
        run.run_id,
        payload={"rows": 12},
        metadata={"kind": "sql_result"},
    )
    summary_artifact = _register_file_artifact(
        services,
        run.run_id,
        payload=_node_summary_payload(),
        metadata={
            "kind": "node_summary",
            "source_artifact_ids": [source_artifact.artifact_id],
        },
    )
    services.run_service.append_event(
        run.run_id,
        "agent.completed",
        "SQL Agent completed",
        node_name="sql_agent",
        artifact_ids=[source_artifact.artifact_id],
        metadata={
            "node_id": "node_legacy",
            "summary": {
                "agent": "sql_agent",
                "artifact_ids": [source_artifact.artifact_id],
            },
        },
    )

    response = client.get(
        f"/agent-runs/{run.run_id}/nodes/node_legacy/summary",
        headers=_user_header(),
    )

    assert response.status_code == 200
    assert response.json()["summary_artifact_id"] == summary_artifact.artifact_id


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


def test_resume_clarification_claims_run_and_schedules_same_thread(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path)
    run = _waiting_clarification_run(services)
    app = create_app(services=services)
    client = TestClient(app)

    monkeypatch.setattr("backend.auth.deps.Auth.ENABLED", False)
    owned_session = SimpleNamespace(id="sess_001", session_db="session_db", mart_db="mart_db")
    ownership: dict[str, str] = {}

    def fake_get_owned_session(session_id: str, username: str):
        ownership.update(session_id=session_id, username=username)
        return owned_session

    seen: dict[str, object] = {}

    def fake_resume(**kwargs) -> None:
        seen.update(kwargs)

    monkeypatch.setattr("backend.agent_runs.routes.get_owned_session", fake_get_owned_session)
    monkeypatch.setattr("backend.agent_runs.routes.resume_agent_run", fake_resume)

    response = client.post(
        f"/agent-runs/{run.run_id}/resume",
        json={"type": "clarification", "answer": "  월별 기준으로 분석해줘  "},
        headers=_user_header(),
    )

    assert response.status_code == 202
    assert response.json() == {
        "run_id": run.run_id,
        "thread_id": "thread_clarification",
        "status": "running",
        "resume_type": "clarification",
    }
    assert ownership == {"session_id": "sess_001", "username": "dev"}
    assert seen["run_id"] == run.run_id
    assert seen["thread_id"] == "thread_clarification"
    assert seen["answer"] == "월별 기준으로 분석해줘"
    assert services.run_service.get_run(run.run_id).status.value == "running"


def test_resume_clarification_rejects_duplicate_submission(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path)
    run = _waiting_clarification_run(services)
    app = create_app(services=services)
    client = TestClient(app)

    monkeypatch.setattr("backend.auth.deps.Auth.ENABLED", False)
    monkeypatch.setattr(
        "backend.agent_runs.routes.get_owned_session",
        lambda session_id, username: SimpleNamespace(id=session_id, session_db="session_db"),
    )
    calls: list[str] = []
    monkeypatch.setattr(
        "backend.agent_runs.routes.resume_agent_run",
        lambda **kwargs: calls.append(kwargs["run_id"]),
    )

    first = client.post(
        f"/agent-runs/{run.run_id}/resume",
        json={"type": "clarification", "answer": "월별"},
        headers=_user_header(),
    )
    second = client.post(
        f"/agent-runs/{run.run_id}/resume",
        json={"type": "clarification", "answer": "분기별"},
        headers=_user_header(),
    )

    assert first.status_code == 202
    assert second.status_code == 409
    assert calls == [run.run_id]


def _succeeded_run(services, *, thread_id: str = "thread_branch_src"):
    run = services.run_service.create_run(
        thread_id=thread_id,
        project_id="sess_001",
        metadata={"session_id": "sess_001", "query": "지난 3개월 매출 추이를 분석해줘"},
    )
    services.run_service.update_status(run.run_id, "running")
    return services.run_service.update_status(run.run_id, "succeeded")


def test_branch_run_starts_background_task_with_plan_from_checkpoint(tmp_path, monkeypatch) -> None:
    from backend.agent_runs.service import BranchPlan

    services = _services(tmp_path)
    run = _succeeded_run(services)
    app = create_app(services=services)
    client = TestClient(app)

    monkeypatch.setattr("backend.auth.deps.Auth.ENABLED", False)
    monkeypatch.setattr(
        "backend.agent_runs.routes.get_owned_session",
        lambda session_id, username: SimpleNamespace(id=session_id, session_db="session_db"),
    )
    plan = BranchPlan(
        upstream_artifact_ids={"sql_agent": ["art_sql_1"]},
        original_question="지난 3개월 매출 추이를 분석해줘",
        target_table="analytics.mart_sales",
        default_parent_node_id="node_1",
    )
    monkeypatch.setattr(
        "backend.agent_runs.routes.prepare_branch_plan",
        lambda **kwargs: plan,
    )
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        "backend.agent_runs.routes.run_branch_task",
        lambda **kwargs: seen.update(kwargs),
    )

    response = client.post(
        f"/agent-runs/{run.run_id}/branch",
        json={"start_stage": "eda", "instruction": "표본이 30 미만인 판매자는 제외하고 다시 분석해줘"},
        headers=_user_header(),
    )

    assert response.status_code == 202
    body = response.json()
    assert body["run_id"] != run.run_id
    assert body["run_id"].startswith("run_")
    assert body["thread_id"] == "thread_branch_src"
    assert body["status"] == "created"
    assert body["start_stage"] == "eda"
    assert body["source_run_id"] == run.run_id
    assert seen["run_id"] == body["run_id"]
    assert seen["thread_id"] == "thread_branch_src"
    assert seen["start_stage"] == "eda"
    assert seen["upstream_artifact_ids"] == {"sql_agent": ["art_sql_1"]}
    assert seen["target_table"] == "analytics.mart_sales"
    # 프론트가 parent_node_id를 안 보내면 체크포인트의 기본값을 쓴다.
    assert seen["parent_node_id"] == "node_1"

    # 원본 run은 이미 종료 상태라 그대로 succeeded로 남는다(재사용하지 않음).
    assert services.run_service.get_run(run.run_id).status.value == "succeeded"
    branch_run_record = services.run_service.get_run(body["run_id"])
    assert branch_run_record.metadata["branched_from_run_id"] == run.run_id


def test_branch_run_prefers_explicit_parent_node_id(tmp_path, monkeypatch) -> None:
    from backend.agent_runs.service import BranchPlan

    services = _services(tmp_path)
    run = _succeeded_run(services)
    app = create_app(services=services)
    client = TestClient(app)

    monkeypatch.setattr("backend.auth.deps.Auth.ENABLED", False)
    monkeypatch.setattr(
        "backend.agent_runs.routes.get_owned_session",
        lambda session_id, username: SimpleNamespace(id=session_id, session_db="session_db"),
    )
    plan = BranchPlan(
        upstream_artifact_ids={},
        original_question="질문",
        target_table=None,
        default_parent_node_id="node_default",
    )
    monkeypatch.setattr("backend.agent_runs.routes.prepare_branch_plan", lambda **kwargs: plan)
    seen: dict[str, object] = {}
    monkeypatch.setattr("backend.agent_runs.routes.run_branch_task", lambda **kwargs: seen.update(kwargs))

    response = client.post(
        f"/agent-runs/{run.run_id}/branch",
        json={"start_stage": "sql", "instruction": "지시", "parent_node_id": "node_clicked"},
        headers=_user_header(),
    )

    assert response.status_code == 202
    assert seen["parent_node_id"] == "node_clicked"


def test_branch_run_rejects_non_succeeded_run(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path)
    run = services.run_service.create_run(
        thread_id="thread_running",
        project_id="sess_001",
        metadata={"session_id": "sess_001"},
    )
    services.run_service.update_status(run.run_id, "running")
    app = create_app(services=services)
    client = TestClient(app)

    monkeypatch.setattr("backend.auth.deps.Auth.ENABLED", False)
    monkeypatch.setattr(
        "backend.agent_runs.routes.get_owned_session",
        lambda session_id, username: SimpleNamespace(id=session_id, session_db="session_db"),
    )

    response = client.post(
        f"/agent-runs/{run.run_id}/branch",
        json={"start_stage": "eda", "instruction": "지시"},
        headers=_user_header(),
    )

    assert response.status_code == 409


def test_branch_run_rejects_missing_upstream_artifacts(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path)
    run = _succeeded_run(services)
    app = create_app(services=services)
    client = TestClient(app)

    monkeypatch.setattr("backend.auth.deps.Auth.ENABLED", False)
    monkeypatch.setattr(
        "backend.agent_runs.routes.get_owned_session",
        lambda session_id, username: SimpleNamespace(id=session_id, session_db="session_db"),
    )

    def fake_prepare_branch_plan(**kwargs):
        from backend.agent_runs.service import BranchPlanError

        raise BranchPlanError("eda 단계 결과가 없어 분기를 시작할 수 없습니다.")

    monkeypatch.setattr("backend.agent_runs.routes.prepare_branch_plan", fake_prepare_branch_plan)

    response = client.post(
        f"/agent-runs/{run.run_id}/branch",
        json={"start_stage": "analysis", "instruction": "지시"},
        headers=_user_header(),
    )

    assert response.status_code == 409
    # 실패 시 run 상태를 running으로 바꾸면 안 된다(분기가 시작되지 않았으므로).
    assert services.run_service.get_run(run.run_id).status.value == "succeeded"


def test_branch_run_rejects_blank_instruction(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path)
    run = _succeeded_run(services)
    app = create_app(services=services)
    client = TestClient(app)

    monkeypatch.setattr("backend.auth.deps.Auth.ENABLED", False)

    response = client.post(
        f"/agent-runs/{run.run_id}/branch",
        json={"start_stage": "eda", "instruction": "   "},
        headers=_user_header(),
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("interrupt_type", "answer", "expected_status"),
    [
        ("analysis_review", "승인", 409),
        ("clarification", "   ", 422),
    ],
)
def test_resume_clarification_validates_waiting_contract(
    tmp_path,
    monkeypatch,
    interrupt_type: str,
    answer: str,
    expected_status: int,
) -> None:
    services = _services(tmp_path)
    run = _waiting_clarification_run(services, interrupt_type=interrupt_type)
    app = create_app(services=services)
    client = TestClient(app)

    monkeypatch.setattr("backend.auth.deps.Auth.ENABLED", False)
    monkeypatch.setattr(
        "backend.agent_runs.routes.get_owned_session",
        lambda session_id, username: SimpleNamespace(id=session_id, session_db="session_db"),
    )

    response = client.post(
        f"/agent-runs/{run.run_id}/resume",
        json={"type": "clarification", "answer": answer},
        headers=_user_header(),
    )

    assert response.status_code == expected_status

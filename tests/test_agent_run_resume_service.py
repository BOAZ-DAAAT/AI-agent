from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

from backend.agent_runs.service import _append_terminal_event, resume_agent_run
from DATA_Analyst_Assistant_Agent.shared.contracts import (
    OrchestrationState,
    SupervisorRunResult,
    SupervisorTerminalState,
)
from data_agent_backend.config import BackendConfig
from data_agent_backend.services.factory import create_backend_services


def _services(tmp_path):
    return create_backend_services(BackendConfig(base_data_dir=tmp_path / ".data_agent"))


def test_resume_agent_run_uses_existing_thread_checkpoint_and_emits_resumed_event(
    tmp_path,
    monkeypatch,
) -> None:
    services = _services(tmp_path)
    run = services.run_service.create_run(
        thread_id="thread_resume",
        project_id="sess_001",
        metadata={"interrupt_type": "clarification", "node": "collect_clarification"},
    )
    services.run_service.update_status(run.run_id, "running")
    session = SimpleNamespace(id="sess_001", session_db="session_db")
    seen: dict[str, object] = {}

    class FakeAdapter:
        def __init__(self, *, services, session, catalog_summary) -> None:
            self.base_data_dir = tmp_path / ".data_agent"

    class FakeSupervisor:
        def __init__(self, adapter, checkpoint_path) -> None:
            seen["checkpoint_path"] = checkpoint_path

        def resume(self, thread_id, payload):
            seen["thread_id"] = thread_id
            seen["payload"] = payload
            return {"terminal_state": "running"}

    monkeypatch.setattr("backend.agent_runs.service.build_session_catalog_summary", lambda session: {"orders": {}})
    monkeypatch.setattr("backend.agent_runs.service.SessionBoundBackendAdapter", FakeAdapter)
    monkeypatch.setattr("backend.agent_runs.service.SupervisorAgent", FakeSupervisor)
    monkeypatch.setattr("backend.agent_runs.service.bind_session_database", lambda session: nullcontext())

    resume_agent_run(
        services=services,
        session=session,
        resume_payload={"answer": "월별 기준"},
        run_id=run.run_id,
        thread_id="thread_resume",
    )

    assert seen["thread_id"] == "thread_resume"
    assert seen["payload"] == {"answer": "월별 기준"}
    assert seen["checkpoint_path"] == tmp_path / ".data_agent" / "thread_resume.sqlite"
    event = services.run_service.list_events(run.run_id)[0]
    assert event.event_type == "human_input.resumed"
    assert event.node_name == "collect_clarification"
    assert event.metadata == {
        "interrupt_type": "clarification",
        "thread_id": "thread_resume",
        "node": "collect_clarification",
    }
    assert "월별 기준" not in event.message
    assert "answer" not in event.metadata


def test_resume_agent_run_analysis_review_passes_selection_payload(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path)
    run = services.run_service.create_run(
        thread_id="thread_review",
        project_id="sess_001",
        metadata={"interrupt_type": "analysis_review", "node": "collect_analysis_review", "resumed_from": "analysis_review"},
    )
    services.run_service.update_status(run.run_id, "running")
    session = SimpleNamespace(id="sess_001", session_db="session_db")
    seen: dict[str, object] = {}

    class FakeAdapter:
        def __init__(self, *, services, session, catalog_summary) -> None:
            self.base_data_dir = tmp_path / ".data_agent"

    class FakeSupervisor:
        def __init__(self, adapter, checkpoint_path) -> None:
            pass

        def resume(self, thread_id, payload):
            seen["payload"] = payload
            return {"terminal_state": "running"}

    monkeypatch.setattr("backend.agent_runs.service.build_session_catalog_summary", lambda session: {"orders": {}})
    monkeypatch.setattr("backend.agent_runs.service.SessionBoundBackendAdapter", FakeAdapter)
    monkeypatch.setattr("backend.agent_runs.service.SupervisorAgent", FakeSupervisor)
    monkeypatch.setattr("backend.agent_runs.service.bind_session_database", lambda session: nullcontext())

    resume_agent_run(
        services=services,
        session=session,
        resume_payload={"approval_id": "run_x:analysis_agent:approval", "selected_option_id": "opt_1", "free_text": None},
        run_id=run.run_id,
        thread_id="thread_review",
    )

    assert seen["payload"] == {
        "approval_id": "run_x:analysis_agent:approval",
        "selected_option_id": "opt_1",
        "free_text": None,
    }
    event = services.run_service.list_events(run.run_id)[0]
    assert event.event_type == "human_input.resumed"
    assert event.node_name == "collect_analysis_review"
    assert event.metadata["interrupt_type"] == "analysis_review"


def test_resume_agent_run_approval_passes_approved_true(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path)
    run = services.run_service.create_run(
        thread_id="thread_approval",
        project_id="sess_001",
        metadata={"resumed_from": "approval"},
    )
    services.run_service.update_status(run.run_id, "running")
    session = SimpleNamespace(id="sess_001", session_db="session_db")
    seen: dict[str, object] = {}

    class FakeAdapter:
        def __init__(self, *, services, session, catalog_summary) -> None:
            self.base_data_dir = tmp_path / ".data_agent"

    class FakeSupervisor:
        def __init__(self, adapter, checkpoint_path) -> None:
            pass

        def resume(self, thread_id, payload):
            seen["payload"] = payload
            return {"terminal_state": "running"}

    monkeypatch.setattr("backend.agent_runs.service.build_session_catalog_summary", lambda session: {"orders": {}})
    monkeypatch.setattr("backend.agent_runs.service.SessionBoundBackendAdapter", FakeAdapter)
    monkeypatch.setattr("backend.agent_runs.service.SupervisorAgent", FakeSupervisor)
    monkeypatch.setattr("backend.agent_runs.service.bind_session_database", lambda session: nullcontext())

    resume_agent_run(
        services=services,
        session=session,
        resume_payload={"approved": True},
        run_id=run.run_id,
        thread_id="thread_approval",
    )

    assert seen["payload"] == {"approved": True}
    event = services.run_service.list_events(run.run_id)[0]
    assert event.event_type == "human_input.resumed"
    # 승인 대기는 노드 이름을 metadata에 안 남기니 기본값(supervisor)으로 떨어진다.
    assert event.node_name == "supervisor"
    assert event.metadata["interrupt_type"] == "approval"


def test_terminal_event_matches_supervisor_terminal_state(tmp_path) -> None:
    services = _services(tmp_path)

    completed_run = services.run_service.create_run(thread_id="thread_completed")
    failed_run = services.run_service.create_run(thread_id="thread_failed")
    completed = SupervisorRunResult(
        kind="state",
        state=OrchestrationState(
            run_id=completed_run.run_id,
            thread_id="thread_completed",
            user_query="매출 분석",
            final_answer="분석 완료",
            terminal_state=SupervisorTerminalState.completed,
        ),
    )
    failed = SupervisorRunResult(
        kind="state",
        state=OrchestrationState(
            run_id=failed_run.run_id,
            thread_id="thread_failed",
            user_query="매출 분석",
            terminal_state=SupervisorTerminalState.failed_terminal,
        ),
    )

    _append_terminal_event(services, completed_run.run_id, completed)
    _append_terminal_event(services, failed_run.run_id, failed)

    assert services.run_service.list_events(completed_run.run_id)[0].event_type == "run.completed"
    assert services.run_service.list_events(failed_run.run_id)[0].event_type == "run.failed"

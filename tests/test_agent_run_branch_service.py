from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from backend.agent_runs.service import (
    BranchPlanError,
    prepare_branch_plan,
    run_branch_task,
)
from DATA_Analyst_Assistant_Agent.supervisor.branch import BranchResult
from data_agent_backend.config import BackendConfig
from data_agent_backend.services.factory import create_backend_services


def _services(tmp_path):
    return create_backend_services(BackendConfig(base_data_dir=tmp_path / ".data_agent"))


_CHECKPOINT_STATE = {
    "current_run_id": "run_branch_001",
    "thread_id": "thread_branch",
    "latest_user_query": "지난 3개월 매출 추이를 분석해줘",
    "analysis_plan": {"target_table": "analytics.mart_sales", "goal": "지난 3개월 매출 추이"},
    "agent_results": [
        {"agent": "sql_agent", "artifact_ids": ["art_sql_1"]},
        {"agent": "eda_agent", "artifact_ids": ["art_eda_1"]},
    ],
    "accepted_evidence": {},
    "state_schema_version": 2,
    "last_completed_node_id": "node_2",
}


class FakeSupervisorForPlan:
    def __init__(self, adapter, checkpoint_path) -> None:
        self.checkpoint_path = checkpoint_path

    def get_checkpoint_state(self, thread_id):
        return _CHECKPOINT_STATE if thread_id == "thread_branch" else None


def test_prepare_branch_plan_collects_upstream_artifacts_and_target_table(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path)
    run = services.run_service.create_run(thread_id="thread_branch", project_id="sess_001")
    monkeypatch.setattr("backend.agent_runs.service.SupervisorAgent", FakeSupervisorForPlan)

    plan = prepare_branch_plan(services=services, run=run, start_stage="analysis")

    assert plan.upstream_artifact_ids == {"sql_agent": ["art_sql_1"], "eda_agent": ["art_eda_1"]}
    assert plan.original_question == "지난 3개월 매출 추이를 분석해줘"
    assert plan.target_table == "analytics.mart_sales"
    assert plan.default_parent_node_id == "node_2"


def test_prepare_branch_plan_first_stage_needs_no_upstream(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path)
    run = services.run_service.create_run(thread_id="thread_branch", project_id="sess_001")
    monkeypatch.setattr("backend.agent_runs.service.SupervisorAgent", FakeSupervisorForPlan)

    plan = prepare_branch_plan(services=services, run=run, start_stage="sql")

    assert plan.upstream_artifact_ids == {}


def test_prepare_branch_plan_raises_when_checkpoint_missing(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path)
    run = services.run_service.create_run(thread_id="thread_unknown", project_id="sess_001")
    monkeypatch.setattr("backend.agent_runs.service.SupervisorAgent", FakeSupervisorForPlan)

    with pytest.raises(BranchPlanError):
        prepare_branch_plan(services=services, run=run, start_stage="eda")


def test_prepare_branch_plan_raises_when_upstream_stage_has_no_artifacts(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path)
    run = services.run_service.create_run(thread_id="thread_branch", project_id="sess_001")
    monkeypatch.setattr("backend.agent_runs.service.SupervisorAgent", FakeSupervisorForPlan)

    # analysis_agent 결과가 체크포인트에 없는데 insight부터 분기하려 하면 막혀야 한다.
    with pytest.raises(BranchPlanError):
        prepare_branch_plan(services=services, run=run, start_stage="insight")


def test_prepare_branch_plan_raises_without_thread_id(tmp_path) -> None:
    services = _services(tmp_path)
    run = services.run_service.create_run(project_id="sess_001")

    with pytest.raises(BranchPlanError):
        prepare_branch_plan(services=services, run=run, start_stage="sql")


def test_run_branch_task_success_marks_run_succeeded_with_new_artifacts(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path)
    run = services.run_service.create_run(thread_id="thread_branch", project_id="sess_001")
    session = SimpleNamespace(id="sess_001", session_db="session_db")
    seen: dict[str, object] = {}

    class FakeAdapter:
        def __init__(self, *, services, session, catalog_summary) -> None:
            pass

    def fake_branch_from(start_stage, instruction, **kwargs):
        seen["start_stage"] = start_stage
        seen["instruction"] = instruction
        seen.update(kwargs)
        return BranchResult(artifact_ids={"eda_agent": ["art_eda_branch_1"]}, last_node_id="node_3")

    monkeypatch.setattr("backend.agent_runs.service.build_session_catalog_summary", lambda session: {"orders": {}})
    monkeypatch.setattr("backend.agent_runs.service.SessionBoundBackendAdapter", FakeAdapter)
    monkeypatch.setattr("backend.agent_runs.service.bind_session_database", lambda session: nullcontext())
    monkeypatch.setattr("backend.agent_runs.service.branch_from", fake_branch_from)

    run_branch_task(
        services=services,
        session=session,
        run_id=run.run_id,
        thread_id="thread_branch",
        start_stage="eda",
        instruction="표본이 30 미만인 판매자는 제외하고 다시 분석해줘",
        upstream_artifact_ids={"sql_agent": ["art_sql_1"]},
        original_question="지난 3개월 매출 추이를 분석해줘",
        target_table="analytics.mart_sales",
        parent_node_id="node_1",
    )

    assert seen["start_stage"] == "eda"
    assert seen["upstream_artifact_ids"] == {"sql_agent": ["art_sql_1"]}
    assert seen["parent_node_id"] == "node_1"
    assert seen["target_table"] == "analytics.mart_sales"

    updated_run = services.run_service.get_run(run.run_id)
    assert updated_run.status.value == "succeeded"
    events = services.run_service.list_events(run.run_id)
    assert events[0].event_type == "branch.started"
    assert events[-1].event_type == "run.completed"
    assert events[-1].metadata["new_artifact_ids"] == {"eda_agent": ["art_eda_branch_1"]}


def test_run_branch_task_agent_failure_marks_run_failed(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path)
    run = services.run_service.create_run(thread_id="thread_branch", project_id="sess_001")
    session = SimpleNamespace(id="sess_001", session_db="session_db")

    class FakeAdapter:
        def __init__(self, *, services, session, catalog_summary) -> None:
            pass

    def fake_branch_from(start_stage, instruction, **kwargs):
        return BranchResult(failed_agent="eda_agent", failure_reason="DB 연결 실패", last_node_id="node_1")

    monkeypatch.setattr("backend.agent_runs.service.build_session_catalog_summary", lambda session: {"orders": {}})
    monkeypatch.setattr("backend.agent_runs.service.SessionBoundBackendAdapter", FakeAdapter)
    monkeypatch.setattr("backend.agent_runs.service.bind_session_database", lambda session: nullcontext())
    monkeypatch.setattr("backend.agent_runs.service.branch_from", fake_branch_from)

    run_branch_task(
        services=services,
        session=session,
        run_id=run.run_id,
        thread_id="thread_branch",
        start_stage="eda",
        instruction="지시사항",
        upstream_artifact_ids={"sql_agent": ["art_sql_1"]},
        original_question="질문",
        target_table=None,
        parent_node_id=None,
    )

    updated_run = services.run_service.get_run(run.run_id)
    assert updated_run.status.value == "failed"
    assert updated_run.metadata["branch_failed_agent"] == "eda_agent"
    events = services.run_service.list_events(run.run_id)
    assert events[-1].event_type == "run.failed"


def test_run_branch_task_exception_marks_run_failed_and_reraises(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path)
    run = services.run_service.create_run(thread_id="thread_branch", project_id="sess_001")
    session = SimpleNamespace(id="sess_001", session_db="session_db")

    def raise_error(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("backend.agent_runs.service.build_session_catalog_summary", raise_error)

    with pytest.raises(RuntimeError):
        run_branch_task(
            services=services,
            session=session,
            run_id=run.run_id,
            thread_id="thread_branch",
            start_stage="eda",
            instruction="지시사항",
            upstream_artifact_ids={"sql_agent": ["art_sql_1"]},
            original_question="질문",
            target_table=None,
            parent_node_id=None,
        )

    updated_run = services.run_service.get_run(run.run_id)
    assert updated_run.status.value == "failed"

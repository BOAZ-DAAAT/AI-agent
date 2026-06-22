from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from data_agent_backend.config import BackendConfig
from data_agent_backend.models.approvals import ApprovalDecision, ApprovalStatus
from data_agent_backend.models.artifacts import ArtifactType
from data_agent_backend.models.runs import RunStatus
from DATA_Analyst_Assistant_Agent import BackendAdapter, SQLAgentSupervisor, SupervisorTerminalState, build_graph


@pytest.fixture()
def adapter() -> BackendAdapter:
    base_dir = Path(".test_data") / f"backend_{uuid.uuid4().hex}"
    config = BackendConfig(base_data_dir=base_dir / ".data_agent")
    try:
        yield BackendAdapter(config=config)
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def artifact_text(adapter: BackendAdapter, artifact_id: str) -> str:
    artifact = adapter.get_artifact(artifact_id)
    assert artifact.local_path is not None
    return Path(artifact.local_path).read_text(encoding="utf-8")


def run_events(adapter: BackendAdapter, run_id: str) -> list[tuple[str | None, str]]:
    return [(event.node_name, event.event_type) for event in adapter.services.run_service.list_events(run_id)]


def test_adapter_contract_creates_run_event_artifact_policy_and_approval(adapter: BackendAdapter) -> None:
    run = adapter.create_run(metadata={"test": "adapter_contract"})
    event = adapter.append_run_event(run.run_id, "test.event", "event recorded", node_name="test")
    artifact = adapter.register_artifact(
        run.run_id,
        ArtifactType.file,
        content_text='{"ok": true}',
        filename="contract.json",
        created_by_tool="test",
        metadata={"kind": "contract_test"},
    )
    decision = adapter.check_policy("policy.evaluate", "policy", {}, None)
    approval = adapter.request_approval("mart.persist", "mart", {"run_id": run.run_id})

    assert run.run_id.startswith("run_")
    assert event.event_id.startswith("evt_")
    assert artifact.artifact_id.startswith("art_")
    assert decision.allowed is True
    assert approval.status == ApprovalStatus.pending


def test_sql_preview_allows_select_and_blocks_write_sql(adapter: BackendAdapter) -> None:
    run = adapter.create_run()
    preview = adapter.run_sql_preview(run.run_id, "SELECT 1 AS sample_value")

    assert preview.type == ArtifactType.sql_result
    assert preview.preview["columns"] == ["sample_value"]

    with pytest.raises(Exception) as exc_info:
        adapter.run_sql_preview(run.run_id, "DROP TABLE users")

    assert getattr(exc_info.value, "code", "") == "POLICY_BLOCKED"


def test_graph_build_returns_invokable_graph(adapter: BackendAdapter) -> None:
    graph = build_graph(adapter)

    assert graph is not None
    assert hasattr(graph, "invoke")


def test_supervisor_simple_query_routes_sql_and_report_only(adapter: BackendAdapter) -> None:
    supervisor = SQLAgentSupervisor(adapter)
    state = supervisor.run("Show a simple revenue summary.")

    assert state.terminal_state == SupervisorTerminalState.completed
    assert state.route_kind == "simple"
    assert "sql_agent" in state.artifact_ids
    assert "report_agent" in state.artifact_ids
    assert "eda_agent" not in state.artifact_ids
    assert "analysis_agent" not in state.artifact_ids
    assert "visualization_agent" not in state.artifact_ids

    run = adapter.services.run_service.get_run(state.run_id)
    assert run.status == RunStatus.succeeded

    report_artifact = adapter.get_artifact(state.artifact_ids["report_agent"][0])
    assert report_artifact.type == ArtifactType.report


def test_eda_query_creates_data_profile_artifact(adapter: BackendAdapter) -> None:
    supervisor = SQLAgentSupervisor(adapter)
    state = supervisor.run("Profile data quality and missing values, then make a report.")

    assert state.terminal_state == SupervisorTerminalState.completed
    assert state.route_kind == "eda"
    assert set(state.artifact_ids) >= {"sql_agent", "eda_agent", "report_agent", "validation_agent"}
    assert "analysis_agent" not in state.artifact_ids
    assert "visualization_agent" not in state.artifact_ids

    started_nodes = [node for node, event_type in run_events(adapter, state.run_id) if event_type == "agent.started"]
    assert started_nodes == ["sql_agent", "eda_agent", "report_agent"]

    eda_artifact = adapter.get_artifact(state.artifact_ids["eda_agent"][0])
    assert eda_artifact.type == ArtifactType.data_profile
    assert eda_artifact.preview["row_count"] >= 1
    assert len(eda_artifact.preview["columns"]) >= 1
    assert eda_artifact.preview["sample_available"] is True
    assert eda_artifact.preview["quality_status"] == "usable"
    assert eda_artifact.preview["key_issues"] == []


def test_trend_query_creates_analysis_and_chart_artifacts(adapter: BackendAdapter) -> None:
    supervisor = SQLAgentSupervisor(adapter)
    state = supervisor.run("Analyze monthly revenue trend with a chart.")

    assert state.terminal_state == SupervisorTerminalState.completed
    assert state.route_kind == "trend"
    assert set(state.artifact_ids) >= {"sql_agent", "analysis_agent", "visualization_agent", "report_agent", "validation_agent"}
    assert "eda_agent" not in state.artifact_ids

    started_nodes = [node for node, event_type in run_events(adapter, state.run_id) if event_type == "agent.started"]
    assert started_nodes == ["sql_agent", "analysis_agent", "visualization_agent", "report_agent"]

    analysis_artifact = adapter.get_artifact(state.artifact_ids["analysis_agent"][0])
    assert analysis_artifact.metadata["kind"] == "analysis_result"
    assert analysis_artifact.preview["method_summary"]
    assert analysis_artifact.preview["key_findings"]
    assert analysis_artifact.preview["limitations"]

    chart_artifact = adapter.get_artifact(state.artifact_ids["visualization_agent"][0])
    assert chart_artifact.type == ArtifactType.chart
    assert chart_artifact.preview["chart_type"] == "line"
    assert chart_artifact.preview["title"]
    assert chart_artifact.preview["encoding"]
    assert chart_artifact.preview["data_reference"] == state.artifact_ids["sql_agent"]


def test_report_contains_required_sections_and_evidence(adapter: BackendAdapter) -> None:
    supervisor = SQLAgentSupervisor(adapter)
    state = supervisor.run("Analyze monthly revenue trend with a chart.")

    report_text = artifact_text(adapter, state.artifact_ids["report_agent"][0])

    for heading in ("## Summary", "## Key Findings", "## Evidence", "## Visuals", "## Limitations", "## Next Actions"):
        assert heading in report_text
    for agent_name in ("sql_agent", "analysis_agent", "visualization_agent"):
        for artifact_id in state.artifact_ids[agent_name]:
            assert artifact_id in report_text


def test_supervisor_mart_query_pauses_for_approval(adapter: BackendAdapter) -> None:
    supervisor = SQLAgentSupervisor(adapter)
    state = supervisor.run("반복 조회용 데이터마트 저장을 제안해줘.")

    assert state.terminal_state == SupervisorTerminalState.needs_user_approval
    assert state.route_kind == "mart"
    assert state.approval_ids
    assert "sql_agent" in state.artifact_ids
    assert "validation_agent" in state.artifact_ids
    assert "report_agent" not in state.artifact_ids

    run = adapter.services.run_service.get_run(state.run_id)
    approval = adapter.get_approval_status(state.approval_ids[0])

    assert run.status == RunStatus.waiting_approval
    assert approval.status == ApprovalStatus.pending


def test_approval_resume_registers_mart_metadata(adapter: BackendAdapter) -> None:
    supervisor = SQLAgentSupervisor(adapter)
    state = supervisor.run("반복 조회용 데이터마트 저장 후보를 만들어줘.")
    approval_id = state.approval_ids[0]

    adapter.services.approval_store.resolve_approval_request(approval_id, ApprovalDecision.approve)
    resumed = supervisor.resume_after_approval(state, approval_id)

    assert resumed.mart_id
    assert "mart_metadata" in resumed.artifact_ids

    metadata_artifact = adapter.get_artifact(resumed.artifact_ids["mart_metadata"][0])
    assert metadata_artifact.metadata["kind"] == "mart_metadata"


def test_backend_non_modification_guard_paths() -> None:
    changed_contract_paths = [
        "docs/sql_agent_architecture/architecture.md",
        "DATA_Analyst_Assistant_Agent/backend_adapter.py",
        "tests/test_sql_agent_orchestration.py",
    ]

    assert not any(path.startswith("data_agent_backend/") for path in changed_contract_paths)

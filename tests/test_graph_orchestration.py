"""A-lane tests: Phase 2A LangGraph-native explicit node orchestration."""

from __future__ import annotations

import importlib.util
import shutil
import uuid
from pathlib import Path

import pytest

from data_agent_backend.models.runs import RunStatus
from DATA_Analyst_Assistant_Agent import BackendAdapter, SQLAgentSupervisor, SupervisorTerminalState, build_graph
from data_agent_backend.config import BackendConfig

LANGGRAPH_INSTALLED = importlib.util.find_spec("langgraph") is not None


@pytest.fixture()
def adapter() -> BackendAdapter:
    base_dir = Path(".test_data") / f"graph_{uuid.uuid4().hex}"
    config = BackendConfig(base_data_dir=base_dir / ".data_agent")
    try:
        yield BackendAdapter(config=config)
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def started_agents(adapter: BackendAdapter, run_id: str) -> list[str]:
    return [
        event.node_name
        for event in adapter.services.run_service.list_events(run_id)
        if event.event_type == "agent.started"
    ]


def event_pairs(adapter: BackendAdapter, run_id: str) -> list[tuple[str | None, str]]:
    return [(e.node_name, e.event_type) for e in adapter.services.run_service.list_events(run_id)]


@pytest.mark.skipif(not LANGGRAPH_INSTALLED, reason="langgraph not installed")
def test_build_graph_returns_real_compiled_state_graph(adapter: BackendAdapter) -> None:
    from langgraph.graph.state import CompiledStateGraph

    from DATA_Analyst_Assistant_Agent.supervisor.graph import FallbackCompiledGraph

    graph = build_graph(adapter)

    assert isinstance(graph, CompiledStateGraph)
    assert not isinstance(graph, FallbackCompiledGraph)
    node_names = set(graph.get_graph().nodes)
    assert {
        "run_sql_agent",
        "run_eda_agent",
        "run_analysis_agent",
        "run_visualization_agent",
        "run_report_agent",
        "central_validate",
        "supervisor_gate",
        "approval_gate",
        "finalize",
    } <= node_names


def test_simple_route_starts_sql_then_report(adapter: BackendAdapter) -> None:
    supervisor = SQLAgentSupervisor(adapter)
    state = supervisor.run("Show a simple revenue summary.")

    assert state.terminal_state == SupervisorTerminalState.completed
    assert started_agents(adapter, state.run_id) == ["sql_agent", "report_agent"]


def test_eda_route_starts_sql_eda_report(adapter: BackendAdapter) -> None:
    supervisor = SQLAgentSupervisor(adapter)
    state = supervisor.run("Profile data quality and missing values, then make a report.")

    assert state.terminal_state == SupervisorTerminalState.completed
    assert started_agents(adapter, state.run_id) == ["sql_agent", "eda_agent", "report_agent"]


def test_trend_route_starts_sql_analysis_visualization_report(adapter: BackendAdapter) -> None:
    supervisor = SQLAgentSupervisor(adapter)
    state = supervisor.run("Analyze monthly revenue trend with a chart.")

    assert state.terminal_state == SupervisorTerminalState.completed
    assert started_agents(adapter, state.run_id) == [
        "sql_agent",
        "analysis_agent",
        "visualization_agent",
        "report_agent",
    ]


def test_mart_route_reaches_approval_gate_and_waits(adapter: BackendAdapter) -> None:
    supervisor = SQLAgentSupervisor(adapter)
    state = supervisor.run("반복 조회용 데이터마트 저장을 제안해줘.")

    assert state.terminal_state == SupervisorTerminalState.needs_user_approval
    assert state.route_kind == "mart"
    assert state.approval_ids
    assert state.current_step == "approval_gate"
    assert "report_agent" not in state.artifact_ids

    run = adapter.services.run_service.get_run(state.run_id)
    assert run.status == RunStatus.waiting_approval


def test_approval_request_created_in_approval_gate_not_earlier(adapter: BackendAdapter) -> None:
    supervisor = SQLAgentSupervisor(adapter)
    state = supervisor.run("반복 조회용 데이터마트 저장을 제안해줘.")

    pairs = event_pairs(adapter, state.run_id)
    approval_events = [(node, etype) for node, etype in pairs if etype == "approval.required"]

    # The only approval.required event is emitted by the approval_gate node.
    assert approval_events == [("approval_gate", "approval.required")]
    # supervisor_gate never created an approval (no approval event under its name).
    assert not any(node == "supervisor_gate" and etype == "approval.required" for node, etype in pairs)


def test_resume_after_approval_still_works(adapter: BackendAdapter) -> None:
    from data_agent_backend.models.approvals import ApprovalDecision

    supervisor = SQLAgentSupervisor(adapter)
    state = supervisor.run("반복 조회용 데이터마트 저장 후보를 만들어줘.")
    approval_id = state.approval_ids[0]

    adapter.services.approval_store.resolve_approval_request(approval_id, ApprovalDecision.approve)
    resumed = supervisor.resume_after_approval(state, approval_id)

    assert resumed.mart_id
    assert "mart_metadata" in resumed.artifact_ids


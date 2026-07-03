from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from data_agent_backend.models.runs import RunStatus
from langgraph.types import Command

from DATA_Analyst_Assistant_Agent.shared.contracts import SupervisorTerminalState
from DATA_Analyst_Assistant_Agent.supervisor.agent import SQLAgentSupervisor, SupervisorAgent


@dataclass
class FakeRun:
    run_id: str


class FakeBackendAdapter:
    base_data_dir = ".data_agent"

    def __init__(self) -> None:
        self.created_runs: list[dict[str, Any]] = []
        self.status_updates: list[tuple[str, Any, dict[str, Any] | None]] = []

    def create_run(self, *, thread_id=None, project_id=None, metadata=None):
        self.created_runs.append(
            {
                "thread_id": thread_id,
                "project_id": project_id,
                "metadata": metadata,
            }
        )
        return FakeRun(run_id="run_001")

    def update_run_status(self, run_id, status, *, metadata=None, context=None):
        self.status_updates.append((run_id, status, metadata))
        return FakeRun(run_id=run_id)


class FakeGraph:
    def invoke(self, state, config):
        return {
            **state,
            "completed_agents": ["sql_agent", "eda_agent", "analysis_agent", "report_agent"],
            "terminal_state": "completed",
            "final_answer": "최종 리포트 생성이 완료되었습니다.",
        }


class CapturingGraph:
    def __init__(self) -> None:
        self.invocations: list[tuple[Any, dict[str, Any]]] = []

    def invoke(self, state, config):
        self.invocations.append((state, config))
        return {"resumed": True}


def _agent_with_terminal(terminal_state: SupervisorTerminalState) -> tuple[FakeBackendAdapter, SupervisorAgent]:
    adapter = FakeBackendAdapter()
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:", use_llm_decision=False)

    def invoke_graph(initial_state, thread_id):
        return {
            **initial_state,
            "terminal_state": terminal_state.value,
            "final_answer": "테스트 종료",
        }

    agent._invoke_graph = invoke_graph  # type: ignore[method-assign]
    return adapter, agent


def test_supervisor_agent_run_returns_orchestration_state(monkeypatch) -> None:
    adapter = FakeBackendAdapter()
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:", use_llm_decision=False)

    monkeypatch.setattr(agent, "_invoke_graph", lambda initial_state, thread_id: FakeGraph().invoke(initial_state, {}))

    state = agent.run("월별 매출 추이를 분석해줘", thread_id="thread_sales_001", datasource_id=None)

    assert state.run_id == "run_001"
    assert state.thread_id == "thread_sales_001"
    assert state.terminal_state.value == "completed"
    assert adapter.status_updates[-1][0] == "run_001"


def test_run_creates_backend_run_with_supervisor_metadata(monkeypatch) -> None:
    adapter = FakeBackendAdapter()
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:", use_llm_decision=False)

    monkeypatch.setattr(agent, "_invoke_graph", lambda initial_state, thread_id: FakeGraph().invoke(initial_state, {}))

    agent.run(
        "월별 매출 추이를 분석해줘",
        thread_id="thread_sales_001",
        datasource_id="datasource_001",
        project_id="project_001",
    )

    assert adapter.created_runs[-1] == {
        "thread_id": "thread_sales_001",
        "project_id": "project_001",
        "metadata": {"query": "월별 매출 추이를 분석해줘", "supervisor": "langgraph"},
    }


def test_completed_terminal_updates_backend_status_succeeded() -> None:
    adapter, agent = _agent_with_terminal(SupervisorTerminalState.completed)

    agent.run("월별 매출 추이를 분석해줘", thread_id="thread_sales_001")

    assert adapter.status_updates[-1][1] == RunStatus.succeeded
    assert adapter.status_updates[-1][2] == {"terminal_state": "completed"}


def test_needs_user_approval_terminal_updates_backend_status_waiting_approval() -> None:
    adapter, agent = _agent_with_terminal(SupervisorTerminalState.needs_user_approval)

    agent.run("월별 매출 추이를 분석해줘", thread_id="thread_sales_001")

    assert adapter.status_updates[-1][1] == RunStatus.waiting_approval
    assert adapter.status_updates[-1][2] == {"terminal_state": "needs_user_approval"}


def test_failed_terminal_updates_backend_status_failed() -> None:
    adapter, agent = _agent_with_terminal(SupervisorTerminalState.failed_terminal)

    agent.run("월별 매출 추이를 분석해줘", thread_id="thread_sales_001")

    assert adapter.status_updates[-1][1] == RunStatus.failed
    assert adapter.status_updates[-1][2] == {"terminal_state": "failed_terminal"}


def test_graph_invoke_exception_updates_backend_status_failed_and_reraises(monkeypatch) -> None:
    adapter = FakeBackendAdapter()
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:", use_llm_decision=False)

    def raise_error(initial_state, thread_id):
        raise RuntimeError("그래프 실패")

    monkeypatch.setattr(agent, "_invoke_graph", raise_error)

    with pytest.raises(RuntimeError, match="그래프 실패"):
        agent.run("월별 매출 추이를 분석해줘", thread_id="thread_sales_001")

    assert adapter.status_updates[-1][1] == RunStatus.failed
    assert adapter.status_updates[-1][2] == {"error": "그래프 실패"}


def test_resume_invokes_graph_with_command_resume(monkeypatch) -> None:
    graph = CapturingGraph()
    agent = SupervisorAgent(FakeBackendAdapter(), checkpoint_path=":memory:", use_llm_decision=False)
    monkeypatch.setattr(agent, "_build_runtime_graph", lambda checkpointer: graph)

    result = agent.resume("thread_sales_001", {"approved": True})

    invoked_state, config = graph.invocations[-1]
    assert result == {"resumed": True}
    assert isinstance(invoked_state, Command)
    assert invoked_state.resume == {"approved": True}
    assert config == {"configurable": {"thread_id": "thread_sales_001"}}


def test_sql_agent_supervisor_alias_points_to_new_supervisor() -> None:
    assert SQLAgentSupervisor is SupervisorAgent

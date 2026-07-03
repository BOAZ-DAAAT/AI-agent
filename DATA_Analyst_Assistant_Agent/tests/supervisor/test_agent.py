from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from data_agent_backend.models.runs import RunStatus
from langgraph.types import Command

from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState, SupervisorTerminalState
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
    def __init__(self, result: dict[str, Any] | Callable[[Any, dict[str, Any]], dict[str, Any]] | None = None) -> None:
        self.result = result or {"resumed": True}
        self.invocations: list[tuple[Any, dict[str, Any]]] = []

    def invoke(self, state, config):
        self.invocations.append((state, config))
        if callable(self.result):
            return self.result(state, config)
        return self.result


class StateSnapshot:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values


class ApprovalResumeGraph(CapturingGraph):
    def __init__(self, checkpoint_state: dict[str, Any], result: dict[str, Any]) -> None:
        super().__init__(result=result)
        self.checkpoint_state = checkpoint_state
        self.state_reads: list[dict[str, Any]] = []
        self.state_updates: list[tuple[dict[str, Any], dict[str, Any], str | None]] = []

    def get_state(self, config):
        self.state_reads.append(config)
        return StateSnapshot(self.checkpoint_state)

    def update_state(self, config, updates, as_node=None):
        self.state_updates.append((config, updates, as_node))
        return {"configurable": {"thread_id": config["configurable"]["thread_id"], "checkpoint": "updated"}}


class CatalogFailingBackendAdapter(FakeBackendAdapter):
    def get_catalog_summary(self, datasource_id):
        raise RuntimeError("카탈로그 조회 실패")


def _completed_graph_state(state: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
    return {
        **state,
        "current_run_id": run_id or state["current_run_id"],
        "completed_agents": ["sql_agent", "eda_agent", "analysis_agent", "report_agent"],
        "terminal_state": "completed",
        "final_answer": "최종 리포트 생성이 완료되었습니다.",
    }


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

    assert isinstance(state, OrchestrationState)
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


def test_needs_clarification_terminal_updates_backend_status_failed() -> None:
    adapter, agent = _agent_with_terminal(SupervisorTerminalState.needs_clarification)

    agent.run("월별 매출 추이를 분석해줘", thread_id="thread_sales_001")

    assert adapter.status_updates[-1][1] == RunStatus.failed
    assert adapter.status_updates[-1][2] == {"terminal_state": "needs_clarification"}


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


def test_catalog_summary_exception_updates_backend_status_failed_and_reraises() -> None:
    adapter = CatalogFailingBackendAdapter()
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:", use_llm_decision=False)

    with pytest.raises(RuntimeError, match="카탈로그 조회 실패"):
        agent.run("월별 매출 추이를 분석해줘", thread_id="thread_sales_001", datasource_id="datasource_001")

    assert adapter.created_runs[-1]["thread_id"] == "thread_sales_001"
    assert adapter.status_updates[-1][0] == "run_001"
    assert adapter.status_updates[-1][1] == RunStatus.failed
    assert adapter.status_updates[-1][2] == {"error": "카탈로그 조회 실패"}


def test_invalid_graph_output_updates_backend_status_failed_and_reraises(monkeypatch) -> None:
    adapter = FakeBackendAdapter()
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:", use_llm_decision=False)

    def invalid_output(initial_state, thread_id):
        return {
            **initial_state,
            "terminal_state": "unknown_terminal",
        }

    monkeypatch.setattr(agent, "_invoke_graph", invalid_output)

    with pytest.raises(ValueError, match="Invalid supervisor terminal_state"):
        agent.run("월별 매출 추이를 분석해줘", thread_id="thread_sales_001")

    assert adapter.status_updates[-1][0] == "run_001"
    assert adapter.status_updates[-1][1] == RunStatus.failed
    assert "Invalid supervisor terminal_state" in adapter.status_updates[-1][2]["error"]


def test_run_uses_runtime_graph_with_thread_config(monkeypatch) -> None:
    graph = CapturingGraph(result=lambda state, config: _completed_graph_state(state))
    agent = SupervisorAgent(FakeBackendAdapter(), checkpoint_path=":memory:", use_llm_decision=False)
    monkeypatch.setattr(agent, "_build_runtime_graph", lambda checkpointer: graph)

    state = agent.run("월별 매출 추이를 분석해줘", thread_id="thread_sales_001")

    invoked_state, config = graph.invocations[-1]
    assert state.terminal_state == SupervisorTerminalState.completed
    assert invoked_state["thread_id"] == "thread_sales_001"
    assert config == {"configurable": {"thread_id": "thread_sales_001"}}


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


def test_decision_model_returns_none_when_model_construction_fails(monkeypatch) -> None:
    agent = SupervisorAgent(FakeBackendAdapter(), checkpoint_path=":memory:")

    def raise_model_error(*args, **kwargs):
        raise RuntimeError("모델 생성 실패")

    monkeypatch.setattr("DATA_Analyst_Assistant_Agent.supervisor.agent.get_chat_model", raise_model_error)

    assert agent._decision_model() is None


def test_resume_consumes_synthetic_approval_and_continues_graph(monkeypatch) -> None:
    adapter = FakeBackendAdapter()
    checkpoint_state = {
        "thread_id": "thread_sales_001",
        "current_run_id": "run_resumed_001",
        "latest_user_query": "월별 매출 추이를 분석해줘",
        "analysis_plan": {},
        "datasource_id": None,
        "catalog_summary": None,
        "retry_counts": {},
        "generated_sql": "",
        "artifacts": {},
        "agent_results": [],
        "completed_agents": [],
        "failed_agents": [],
        "error_state": {},
        "max_retry_per_agent": 1,
        "terminal_state": "needs_user_approval",
        "next_action": "finalize",
        "final_answer": "사용자 승인이 필요합니다.",
        "pending_approval": {
            "approval_id": "run_resumed_001:sql_agent:approval",
            "agent": "sql_agent",
            "reason": "SQL 실행 승인 필요",
            "approval_type": "agent_approval",
        },
    }
    resumed_result = _completed_graph_state(
        {
            **checkpoint_state,
            "pending_approval": None,
            "artifacts": {"report_agent": [{"artifact_id": "artifact_report"}]},
            "agent_results": [
                {
                    "agent": "report_agent",
                    "status": "success",
                    "summary": "리포트 완료",
                    "artifact_ids": ["artifact_report"],
                    "artifacts": [],
                    "validation_errors": [],
                    "validation_warnings": [],
                    "fallback_used": False,
                    "retryable": False,
                    "error": "",
                }
            ],
        },
        run_id="run_resumed_001",
    )
    graph = ApprovalResumeGraph(checkpoint_state, resumed_result)
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:", use_llm_decision=False)
    monkeypatch.setattr(agent, "_build_runtime_graph", lambda checkpointer: graph)

    result = agent.resume("thread_sales_001", {"approved": True})

    assert result["terminal_state"] == "completed"
    assert graph.state_reads == [{"configurable": {"thread_id": "thread_sales_001"}}]
    assert graph.state_updates == [
        (
            {"configurable": {"thread_id": "thread_sales_001"}},
            {
                "pending_approval": None,
                "terminal_state": "running",
                "next_action": "call_sql_agent",
                "final_answer": "",
            },
            "summarize_step",
        )
    ]
    assert graph.invocations == [
        (None, {"configurable": {"thread_id": "thread_sales_001", "checkpoint": "updated"}})
    ]
    assert adapter.status_updates[-1] == (
        "run_resumed_001",
        RunStatus.succeeded,
        {"terminal_state": "completed"},
    )


def test_resume_updates_backend_status_when_graph_returns_terminal_state(monkeypatch) -> None:
    adapter = FakeBackendAdapter()
    graph = CapturingGraph(
        result=lambda state, config: _completed_graph_state(
            {
                "thread_id": "thread_sales_001",
                "current_run_id": "run_resumed_001",
                "latest_user_query": "월별 매출 추이를 분석해줘",
                "analysis_plan": {},
                "datasource_id": None,
                "catalog_summary": None,
                "retry_counts": {},
                "generated_sql": "",
                "artifacts": {},
                "agent_results": [],
                "completed_agents": [],
                "error_state": {},
                "max_retry_per_agent": 1,
            },
            run_id="run_resumed_001",
        )
    )
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:", use_llm_decision=False)
    monkeypatch.setattr(agent, "_build_runtime_graph", lambda checkpointer: graph)

    result = agent.resume("thread_sales_001", {"approved": True})

    assert result["current_run_id"] == "run_resumed_001"
    assert adapter.status_updates[-1] == (
        "run_resumed_001",
        RunStatus.succeeded,
        {"terminal_state": "completed"},
    )


def test_resume_does_not_update_backend_status_without_terminal_state(monkeypatch) -> None:
    adapter = FakeBackendAdapter()
    result = {
        "thread_id": "thread_sales_001",
        "current_run_id": "run_resumed_001",
        "latest_user_query": "월별 매출 추이를 분석해줘",
    }
    graph = CapturingGraph(result=result)
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:", use_llm_decision=False)
    monkeypatch.setattr(agent, "_build_runtime_graph", lambda checkpointer: graph)

    returned = agent.resume("thread_sales_001", {"approved": True})

    assert returned is result
    assert adapter.status_updates == []


def test_resume_does_not_update_backend_status_for_running_terminal_state(monkeypatch) -> None:
    adapter = FakeBackendAdapter()
    result = {
        "thread_id": "thread_sales_001",
        "current_run_id": "run_resumed_001",
        "terminal_state": "running",
        "latest_user_query": "월별 매출 추이를 분석해줘",
    }
    graph = CapturingGraph(result=result)
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:", use_llm_decision=False)
    monkeypatch.setattr(agent, "_build_runtime_graph", lambda checkpointer: graph)

    returned = agent.resume("thread_sales_001", {"approved": True})

    assert returned is result
    assert adapter.status_updates == []


def test_sql_agent_supervisor_alias_points_to_new_supervisor() -> None:
    assert SQLAgentSupervisor is SupervisorAgent

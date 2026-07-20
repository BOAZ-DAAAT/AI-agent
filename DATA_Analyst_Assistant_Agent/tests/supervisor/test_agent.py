from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from data_agent_backend.models.runs import RunStatus
from langgraph.types import Command

from DATA_Analyst_Assistant_Agent.shared.contracts import (
    OrchestrationState,
    SupervisorRunResult,
    SupervisorTerminalState,
)
from DATA_Analyst_Assistant_Agent.supervisor.agent import SQLAgentSupervisor, SupervisorAgent


@dataclass
class FakeRun:
    run_id: str


@dataclass
class FakeInterrupt:
    value: dict[str, Any]


class FakeBackendAdapter:
    base_data_dir = ".data_agent"

    def __init__(self) -> None:
        self.created_runs: list[dict[str, Any]] = []
        self.status_updates: list[tuple[str, Any, dict[str, Any] | None]] = []
        self.events: list[dict[str, Any]] = []
        self.artifact_hashes: dict[str, str] = {}

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

    def append_run_event(
        self,
        run_id,
        event_type,
        message,
        *,
        event_key=None,
        node_name=None,
        tool_name=None,
        artifact_ids=None,
        approval_id=None,
        metadata=None,
        context=None,
    ):
        self.events.append(
            {
                "run_id": run_id,
                "event_type": event_type,
                "message": message,
                "event_key": event_key,
                "node_name": node_name,
                "metadata": metadata,
            }
        )
        return {"event_id": f"evt_{len(self.events)}"}

    def get_artifact(self, artifact_id):
        return type(
            "FakeArtifact",
            (),
            {"artifact_id": artifact_id, "content_hash": self.artifact_hashes[artifact_id]},
        )()


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
    def __init__(self, values: dict[str, Any], interrupts: list[dict[str, Any]] | None = None) -> None:
        self.values = values
        self.tasks = [
            type("FakeTask", (), {"interrupts": tuple(FakeInterrupt(value) for value in interrupts or [])})()
        ] if interrupts is not None else []


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


class CheckpointResumeGraph(CapturingGraph):
    def __init__(
        self,
        checkpoint_state: dict[str, Any],
        result: dict[str, Any],
        *,
        interrupts: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(result=result)
        self.checkpoint_state = checkpoint_state
        self.interrupts = interrupts
        self.state_reads: list[dict[str, Any]] = []

    def get_state(self, config):
        self.state_reads.append(config)
        return StateSnapshot(self.checkpoint_state, self.interrupts)


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
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")

    def invoke_graph(initial_state, thread_id):
        return {
            **initial_state,
            "terminal_state": terminal_state.value,
            "final_answer": "테스트 종료",
        }

    agent._invoke_graph = invoke_graph  # type: ignore[method-assign]
    return adapter, agent


def test_supervisor_agent_constructor_no_longer_exposes_use_llm_decision() -> None:
    assert "use_llm_decision" not in inspect.signature(SupervisorAgent).parameters


def test_supervisor_agent_run_returns_orchestration_state(monkeypatch) -> None:
    adapter = FakeBackendAdapter()
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")

    monkeypatch.setattr(agent, "_invoke_graph", lambda initial_state, thread_id: FakeGraph().invoke(initial_state, {}))

    result = agent.run("월별 매출 추이를 분석해줘", thread_id="thread_sales_001", datasource_id=None)

    assert isinstance(result, SupervisorRunResult)
    assert result.kind == "state"
    assert result.state is not None
    assert result.interrupt is None
    state = result.state
    assert isinstance(state, OrchestrationState)
    assert state.run_id == "run_001"
    assert state.thread_id == "thread_sales_001"
    assert state.terminal_state.value == "completed"
    assert state.final_answer == "최종 리포트 생성이 완료되었습니다."
    assert adapter.status_updates[-1][0] == "run_001"


def test_supervisor_agent_run_preserves_plan_failure_state(monkeypatch) -> None:
    adapter = FakeBackendAdapter()
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")

    def plan_failure(initial_state, thread_id):
        return {
            **initial_state,
            "terminal_state": "failed_terminal",
            "final_answer": "계획 생성에 실패했습니다.",
            "error_state": {"message": "계획 생성 실패", "retryable": False},
        }

    monkeypatch.setattr(agent, "_invoke_graph", plan_failure)

    result = agent.run("월별 매출 추이를 분석해줘", thread_id="thread_sales_001")

    assert result.kind == "state"
    assert result.state is not None
    assert result.state.final_answer == "계획 생성에 실패했습니다."
    assert result.state.plan is None
    assert result.state.generated_sql == ""


def test_supervisor_agent_run_preserves_approval_final_answer(monkeypatch) -> None:
    adapter = FakeBackendAdapter()
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")

    def approval_waiting(initial_state, thread_id):
        return {
            **initial_state,
            "terminal_state": "needs_user_approval",
            "final_answer": "SQL 실행 승인이 필요합니다.",
            "pending_approval": {
                "approval_id": "run_001:sql_agent:approval",
                "agent": "sql_agent",
            },
        }

    monkeypatch.setattr(agent, "_invoke_graph", approval_waiting)

    result = agent.run("월별 매출 추이를 분석해줘", thread_id="thread_sales_001")

    assert result.kind == "state"
    assert result.state is not None
    assert result.state.final_answer == "SQL 실행 승인이 필요합니다."
    assert result.state.approval_ids == ["run_001:sql_agent:approval"]


def test_supervisor_agent_run_keeps_generated_and_source_sql_consistent(monkeypatch) -> None:
    adapter = FakeBackendAdapter()
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")
    generated_sql = "SELECT month, SUM(amount) FROM sales GROUP BY month"

    def completed_with_sql(initial_state, thread_id):
        return {
            **initial_state,
            "analysis_plan": {
                "goal": "월별 매출 추이 분석",
                "source_sql": "SELECT month, amount FROM sales",
            },
            "generated_sql": generated_sql,
            "terminal_state": "completed",
            "final_answer": "월별 매출 분석이 완료되었습니다.",
        }

    monkeypatch.setattr(agent, "_invoke_graph", completed_with_sql)

    result = agent.run("월별 매출 추이를 분석해줘", thread_id="thread_sales_001")

    assert result.kind == "state"
    assert result.state is not None
    assert result.state.plan is not None
    assert result.state.final_answer == "월별 매출 분석이 완료되었습니다."
    assert result.state.generated_sql == generated_sql
    assert result.state.plan.generated_sql == generated_sql
    assert result.state.plan.source_sql == "SELECT month, amount FROM sales"


def test_supervisor_agent_run_returns_interrupt_result_and_waiting_input_status(monkeypatch) -> None:
    adapter = FakeBackendAdapter()
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")

    def interrupt_output(initial_state, thread_id):
        return {
            "__interrupt__": (
                FakeInterrupt(
                    {
                        "type": "clarification",
                        "status": "waiting_input",
                        "run_id": initial_state["current_run_id"],
                        "thread_id": thread_id,
                        "question": "어떤 기간과 단위로 매출을 분석할까요?",
                        "node": "collect_clarification",
                        "expected_resume": {"answer": "string"},
                    }
                ),
            )
        }

    monkeypatch.setattr(agent, "_invoke_graph", interrupt_output)

    result = agent.run("매출", thread_id="thread_sales_001", datasource_id=None)

    assert result.kind == "interrupt"
    assert result.state is None
    assert result.interrupt is not None
    assert result.interrupt.type == "clarification"
    assert result.interrupt.question == "어떤 기간과 단위로 매출을 분석할까요?"
    assert adapter.status_updates[-1] == (
        "run_001",
        RunStatus.waiting_input,
        {"interrupt_type": "clarification", "node": "collect_clarification"},
    )
    assert adapter.events[-1]["event_type"] == "human_input.required"
    assert adapter.events[-1]["node_name"] == "collect_clarification"


def test_run_creates_backend_run_with_supervisor_metadata(monkeypatch) -> None:
    adapter = FakeBackendAdapter()
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")

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
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")

    def raise_error(initial_state, thread_id):
        raise RuntimeError("그래프 실패")

    monkeypatch.setattr(agent, "_invoke_graph", raise_error)

    with pytest.raises(RuntimeError, match="그래프 실패"):
        agent.run("월별 매출 추이를 분석해줘", thread_id="thread_sales_001")

    assert adapter.status_updates[-1][1] == RunStatus.failed
    assert adapter.status_updates[-1][2] == {"error": "그래프 실패"}


def test_catalog_summary_exception_updates_backend_status_failed_and_reraises() -> None:
    adapter = CatalogFailingBackendAdapter()
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")

    with pytest.raises(RuntimeError, match="카탈로그 조회 실패"):
        agent.run("월별 매출 추이를 분석해줘", thread_id="thread_sales_001", datasource_id="datasource_001")

    assert adapter.created_runs[-1]["thread_id"] == "thread_sales_001"
    assert adapter.status_updates[-1][0] == "run_001"
    assert adapter.status_updates[-1][1] == RunStatus.failed
    assert adapter.status_updates[-1][2] == {"error": "카탈로그 조회 실패"}


def test_invalid_graph_output_updates_backend_status_failed_and_reraises(monkeypatch) -> None:
    adapter = FakeBackendAdapter()
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")

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
    agent = SupervisorAgent(FakeBackendAdapter(), checkpoint_path=":memory:")
    monkeypatch.setattr(agent, "_build_runtime_graph", lambda checkpointer: graph)

    result = agent.run("월별 매출 추이를 분석해줘", thread_id="thread_sales_001")

    invoked_state, config = graph.invocations[-1]
    assert result.kind == "state"
    assert result.state is not None
    assert result.state.terminal_state == SupervisorTerminalState.completed
    assert invoked_state["thread_id"] == "thread_sales_001"
    assert config == {"configurable": {"thread_id": "thread_sales_001"}}


def test_resume_invokes_graph_with_command_resume(monkeypatch) -> None:
    graph = CapturingGraph(
        result=_completed_graph_state(
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
    agent = SupervisorAgent(FakeBackendAdapter(), checkpoint_path=":memory:")
    monkeypatch.setattr(agent, "_build_runtime_graph", lambda checkpointer: graph)

    result = agent.resume("thread_sales_001", {"approved": True})

    invoked_state, config = graph.invocations[-1]
    assert result.kind == "state"
    assert result.state is not None
    assert result.state.run_id == "run_resumed_001"
    assert isinstance(invoked_state, Command)
    assert invoked_state.resume == {"approved": True}
    assert config == {"configurable": {"thread_id": "thread_sales_001"}}


def test_resume_with_clarification_answer_uses_command_resume_and_updates_status(monkeypatch) -> None:
    adapter = FakeBackendAdapter()
    checkpoint_state = {
        "thread_id": "thread_sales_001",
        "current_run_id": "run_resumed_001",
        "latest_user_query": "매출",
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
        "terminal_state": "running",
    }
    graph = CheckpointResumeGraph(
        checkpoint_state,
        _completed_graph_state(checkpoint_state, run_id="run_resumed_001"),
        interrupts=[
            {
                "type": "clarification",
                "status": "waiting_input",
                "run_id": "run_resumed_001",
                "thread_id": "thread_sales_001",
                "question": "기간은?",
                "node": "collect_clarification",
                "expected_resume": {"answer": "string"},
            }
        ],
    )
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")
    monkeypatch.setattr(agent, "_build_runtime_graph", lambda checkpointer: graph)

    result = agent.resume("thread_sales_001", {"answer": "최근 6개월 월별 매출"})

    invoked_state, config = graph.invocations[-1]
    assert result.kind == "state"
    assert result.state is not None
    assert result.state.run_id == "run_resumed_001"
    assert isinstance(invoked_state, Command)
    assert invoked_state.resume == {"answer": "최근 6개월 월별 매출"}
    assert config == {"configurable": {"thread_id": "thread_sales_001"}}
    assert adapter.status_updates[0] == ("run_resumed_001", RunStatus.running, {"resumed_from": "clarification"})
    assert adapter.status_updates[-1] == ("run_resumed_001", RunStatus.succeeded, {"terminal_state": "completed"})


def test_resume_with_clarification_answer_without_checkpoint_run_id_raises(monkeypatch) -> None:
    adapter = FakeBackendAdapter()
    graph = CheckpointResumeGraph({}, {"resumed": True})
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")
    monkeypatch.setattr(agent, "_build_runtime_graph", lambda checkpointer: graph)

    with pytest.raises(ValueError, match="thread_id"):
        agent.resume("missing_thread", {"answer": "최근 6개월 월별 매출"})

    assert graph.invocations == []
    assert adapter.status_updates == []


def _analysis_review_checkpoint() -> tuple[dict[str, Any], dict[str, Any]]:
    review_request = {
        "decision_type": "aggregation_method",
        "question": "대표값은?",
        "proposal": "대표값 선택",
        "rationale": ["분포가 비대칭입니다."],
        "evidence": {"sample_size": 10},
        "options": [
            {
                "id": "mean",
                "label": "평균",
                "method": "산술 평균",
                "assumptions": [],
                "advantages": [],
                "limitations": ["극단값 영향"],
                "impact": "평균을 사용합니다.",
                "recommended": False,
            },
            {
                "id": "median",
                "label": "중앙값",
                "method": "50% 분위수",
                "assumptions": [],
                "advantages": ["강건함"],
                "limitations": [],
                "impact": "중앙값을 사용합니다.",
                "recommended": True,
            },
        ],
        "recommended_option_id": "median",
        "allow_free_text": True,
        "free_text_prompt": "다른 제약",
        "impact_if_approved": "선택을 반영합니다.",
        "requires_followup_analysis": True,
    }
    approval_id = "run_resumed_001:analysis_agent:candidate_001:approval"
    checkpoint = {
        "thread_id": "thread_sales_001",
        "current_run_id": "run_resumed_001",
        "latest_user_query": "매출 분석",
        "state_schema_version": 6,
        "pending_result": {
            "candidate_id": "candidate_001",
            "validation_id": "validation_001",
            "content_hashes": {"analysis_001": "hash_001"},
            "result": {"agent": "analysis_agent", "status": "approval_required", "summary": "선택 필요"},
        },
        "pending_approval": {
            "approval_id": approval_id,
            "agent": "analysis_agent",
            "reason": "선택 필요",
            "approval_type": "analysis.review",
            "candidate_id": "candidate_001",
            "validation_id": "validation_001",
            "content_hashes": {"analysis_001": "hash_001"},
            "review_request": review_request,
            "expected_resume": {
                "approval_id": "string",
                "selected_option_id": "string?",
                "free_text": "string?",
            },
        },
        "terminal_state": "running",
        "next_action": "collect_analysis_review",
    }
    interrupt_payload = {
        "type": "analysis_review",
        "status": "waiting_input",
        "run_id": "run_resumed_001",
        "thread_id": "thread_sales_001",
        "question": "대표값은?",
        "node": "collect_analysis_review",
        "expected_resume": checkpoint["pending_approval"]["expected_resume"],
        "approval_id": approval_id,
        "review_request": review_request,
    }
    return checkpoint, interrupt_payload


def test_resume_analysis_review_validates_checkpoint_before_status_update(monkeypatch) -> None:
    checkpoint, interrupt_payload = _analysis_review_checkpoint()
    adapter = FakeBackendAdapter()
    adapter.artifact_hashes["analysis_001"] = "hash_001"
    resumed_result = {**checkpoint, "terminal_state": "running"}
    graph = CheckpointResumeGraph(
        checkpoint,
        resumed_result,
        interrupts=[interrupt_payload],
    )
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")
    monkeypatch.setattr(agent, "_build_runtime_graph", lambda checkpointer: graph)

    returned = agent.resume(
        "thread_sales_001",
        {"approval_id": interrupt_payload["approval_id"], "selected_option_id": " median "},
    )

    command, _ = graph.invocations[-1]
    assert returned is resumed_result
    assert command.resume == {
        "approval_id": interrupt_payload["approval_id"],
        "selected_option_id": "median",
    }
    assert adapter.status_updates == [
        ("run_resumed_001", RunStatus.running, {"resumed_from": "analysis_review"})
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"approval_id": "stale", "selected_option_id": "median"},
        {"approval_id": "run_resumed_001:analysis_agent:candidate_001:approval", "selected_option_id": "unknown"},
        {"approved": True},
        {"answer": "중앙값"},
    ],
)
def test_invalid_analysis_review_resume_leaves_backend_and_graph_unchanged(
    monkeypatch,
    payload: dict[str, Any],
) -> None:
    checkpoint, interrupt_payload = _analysis_review_checkpoint()
    adapter = FakeBackendAdapter()
    adapter.artifact_hashes["analysis_001"] = "hash_001"
    graph = CheckpointResumeGraph(checkpoint, {"resumed": True}, interrupts=[interrupt_payload])
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")
    monkeypatch.setattr(agent, "_build_runtime_graph", lambda checkpointer: graph)

    with pytest.raises(ValueError):
        agent.resume("thread_sales_001", payload)

    assert graph.invocations == []
    assert adapter.status_updates == []
    assert adapter.events == []


def test_analysis_review_resume_rejects_changed_artifact_hash_without_mutation(monkeypatch) -> None:
    checkpoint, interrupt_payload = _analysis_review_checkpoint()
    adapter = FakeBackendAdapter()
    adapter.artifact_hashes["analysis_001"] = "changed"
    graph = CheckpointResumeGraph(checkpoint, {"resumed": True}, interrupts=[interrupt_payload])
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")
    monkeypatch.setattr(agent, "_build_runtime_graph", lambda checkpointer: graph)

    with pytest.raises(ValueError, match="content hash"):
        agent.resume(
            "thread_sales_001",
            {"approval_id": interrupt_payload["approval_id"], "selected_option_id": "median"},
        )

    assert graph.invocations == []
    assert adapter.status_updates == []


def test_clarification_interrupt_rejects_structured_analysis_pending_state(monkeypatch) -> None:
    checkpoint, _ = _analysis_review_checkpoint()
    clarification_interrupt = {
        "type": "clarification",
        "status": "waiting_input",
        "run_id": "run_resumed_001",
        "thread_id": "thread_sales_001",
        "question": "기간은?",
        "node": "collect_clarification",
        "expected_resume": {"answer": "string"},
    }
    adapter = FakeBackendAdapter()
    graph = CheckpointResumeGraph(
        checkpoint,
        {"resumed": True},
        interrupts=[clarification_interrupt],
    )
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")
    monkeypatch.setattr(agent, "_build_runtime_graph", lambda checkpointer: graph)

    with pytest.raises(ValueError, match="불일치"):
        agent.resume("thread_sales_001", {"answer": "최근 6개월"})

    assert graph.invocations == []
    assert adapter.status_updates == []


@pytest.mark.parametrize("payload", [{}, {"answer": ""}, {"answer": "   "}])
def test_resume_with_invalid_clarification_answer_raises_without_invoking_graph(monkeypatch, payload) -> None:
    graph = CapturingGraph()
    agent = SupervisorAgent(FakeBackendAdapter(), checkpoint_path=":memory:")
    monkeypatch.setattr(agent, "_build_runtime_graph", lambda checkpointer: graph)

    with pytest.raises(ValueError, match="answer"):
        agent.resume("thread_sales_001", payload)

    assert graph.invocations == []


def test_decision_model_propagates_model_construction_failure(monkeypatch) -> None:
    agent = SupervisorAgent(FakeBackendAdapter(), checkpoint_path=":memory:")

    def raise_model_error(*args, **kwargs):
        raise RuntimeError("모델 생성 실패")

    monkeypatch.setattr("DATA_Analyst_Assistant_Agent.supervisor.agent.get_chat_model", raise_model_error)

    with pytest.raises(RuntimeError, match="모델 생성 실패"):
        agent._decision_model()


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
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")
    monkeypatch.setattr(agent, "_build_runtime_graph", lambda checkpointer: graph)

    result = agent.resume("thread_sales_001", {"approved": True})

    assert result.kind == "state"
    assert result.state is not None
    assert result.state.terminal_state == SupervisorTerminalState.completed
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
                "decide_next_action",
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
    assert adapter.status_updates[0] == (
        "run_resumed_001",
        RunStatus.running,
        {"resumed_from": "approval"},
    )


@pytest.mark.parametrize("candidate_status", ["success", "approval_required"])
def test_resume_approval_with_matching_candidate_hash_promotes_without_agent_rerun(
    monkeypatch,
    candidate_status: str,
) -> None:
    adapter = FakeBackendAdapter()
    adapter.artifact_hashes["artifact_sql"] = "hash_sql"
    base = {
        "thread_id": "thread_sales_001",
        "current_run_id": "run_resumed_001",
        "latest_user_query": "월별 매출",
        "analysis_plan": {},
        "datasource_id": None,
        "catalog_summary": None,
        "retry_counts": {},
        "generated_sql": "",
        "artifacts": {},
        "accepted_evidence": {},
        "agent_results": [],
        "completed_agents": [],
        "failed_agents": [],
        "error_state": {},
        "max_retry_per_agent": 1,
        "state_schema_version": 2,
        "result_history": [],
        "rejected_results": [],
        "quarantined_artifacts": [],
        "semantic_retry_counts": {},
        "run_events": [],
        "terminal_state": "needs_user_approval",
        "next_action": "finalize",
        "final_answer": "승인 필요",
        "pending_result": {
            "candidate_id": "candidate_001",
            "validation_id": "validation_001",
            "content_hashes": {"artifact_sql": "hash_sql"},
            "state_updates": {"generated_sql": "SELECT 1"},
            "result": {
                "agent": "sql_agent",
                "status": candidate_status,
                "summary": "SQL 완료",
                "artifact_ids": ["artifact_sql"],
                "artifacts": [{"artifact_id": "artifact_sql", "content_hash": "hash_sql"}],
                "approval": {"required": True, "reason": "승인 필요"},
            },
        },
        "pending_approval": {
            "approval_id": "approval_001",
            "agent": "sql_agent",
            "reason": "승인 필요",
            "approval_type": "sql.execute",
            "candidate_id": "candidate_001",
            "validation_id": "validation_001",
            "content_hashes": {"artifact_sql": "hash_sql"},
        },
    }
    graph = ApprovalResumeGraph(base, _completed_graph_state(base, run_id="run_resumed_001"))
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")
    monkeypatch.setattr(agent, "_build_runtime_graph", lambda checkpointer: graph)

    agent.resume("thread_sales_001", {"approved": True})

    _, updates, as_node = graph.state_updates[0]
    assert as_node == "commit_candidate"
    assert updates["pending_result"] is None
    assert updates["completed_agents"] == ["sql_agent"]
    assert updates["agent_results"][-1]["status"] == "success"
    assert updates["generated_sql"] == "SELECT 1"


def test_resume_approval_with_changed_hash_invalidates_and_revalidates(monkeypatch) -> None:
    adapter = FakeBackendAdapter()
    adapter.artifact_hashes["artifact_sql"] = "changed_hash"
    checkpoint = {
        "current_run_id": "run_resumed_001",
        "pending_result": {
            "candidate_id": "candidate_001",
            "validation_id": "validation_001",
            "content_hashes": {"artifact_sql": "old_hash"},
            "state_updates": {},
            "result": {"agent": "sql_agent", "status": "success", "summary": "SQL 완료"},
        },
        "pending_approval": {
            "agent": "sql_agent",
            "candidate_id": "candidate_001",
            "validation_id": "validation_001",
            "content_hashes": {"artifact_sql": "old_hash"},
        },
        "terminal_state": "needs_user_approval",
        "state_schema_version": 2,
        "accepted_evidence": {},
        "result_history": [],
        "rejected_results": [],
        "quarantined_artifacts": [],
        "semantic_retry_counts": {},
        "run_events": [],
    }
    graph = ApprovalResumeGraph(checkpoint, {"current_run_id": "run_resumed_001", "terminal_state": "running"})
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")
    monkeypatch.setattr(agent, "_build_runtime_graph", lambda checkpointer: graph)

    agent.resume("thread_sales_001", {"approved": True})

    _, updates, as_node = graph.state_updates[0]
    assert as_node == "execute_subagent"
    assert updates["pending_approval"] is None
    assert updates["pending_validation"] is None
    assert updates["current_step"] == "execute_subagent"
    assert updates["run_events"][-1]["type"] == "approval.invalidated"


def test_resume_approval_without_pending_approval_raises_when_checkpoint_is_readable(monkeypatch) -> None:
    adapter = FakeBackendAdapter()
    checkpoint_state = {
        "thread_id": "thread_sales_001",
        "current_run_id": "run_resumed_001",
        "latest_user_query": "월별 매출 추이를 분석해줘",
        "terminal_state": "completed",
        "pending_approval": None,
    }
    graph = ApprovalResumeGraph(checkpoint_state, {"resumed": True})
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")
    monkeypatch.setattr(agent, "_build_runtime_graph", lambda checkpointer: graph)

    with pytest.raises(ValueError, match="승인 대기"):
        agent.resume("thread_sales_001", {"approved": True})

    assert graph.invocations == []
    assert graph.state_updates == []
    assert adapter.status_updates == []


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
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")
    monkeypatch.setattr(agent, "_build_runtime_graph", lambda checkpointer: graph)

    result = agent.resume("thread_sales_001", {"approved": True})

    assert result.kind == "state"
    assert result.state is not None
    assert result.state.run_id == "run_resumed_001"
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
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")
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
    agent = SupervisorAgent(adapter, checkpoint_path=":memory:")
    monkeypatch.setattr(agent, "_build_runtime_graph", lambda checkpointer: graph)

    returned = agent.resume("thread_sales_001", {"approved": True})

    assert returned is result
    assert adapter.status_updates == []


def test_sql_agent_supervisor_alias_points_to_new_supervisor() -> None:
    assert SQLAgentSupervisor is SupervisorAgent

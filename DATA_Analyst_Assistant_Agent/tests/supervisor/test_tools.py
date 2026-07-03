from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from data_agent_backend.models.artifacts import ArtifactRef, ArtifactType

from DATA_Analyst_Assistant_Agent.shared.contracts import (
    AgentEnvelope,
    AgentStatus,
    ApprovalRequirement,
    BusinessFlag,
    LocalCheck,
    OrchestrationState,
    RetryHint,
    ValidationBlock,
)
from DATA_Analyst_Assistant_Agent.supervisor.state import empty_supervisor_state
from DATA_Analyst_Assistant_Agent.supervisor import tools
from DATA_Analyst_Assistant_Agent.supervisor.tools import SubAgentAdapter


@dataclass
class FakeArtifactRecord:
    artifact_id: str
    type: str
    uri: str | None = None
    metadata: dict[str, Any] | None = None
    preview: dict[str, Any] | None = None


class FakeAdapter:
    base_data_dir = ".data_agent"

    def get_artifact(self, artifact_id: str) -> FakeArtifactRecord:
        return FakeArtifactRecord(
            artifact_id=artifact_id,
            type="sql_result",
            metadata={"kind": "sql_result"},
            preview={"row_count": 3},
        )


class StubAgent:
    name = "sql_agent"

    def run(self, state: OrchestrationState, runtime) -> AgentEnvelope:
        state.generated_sql = "SELECT 1 AS sample_value"
        return AgentEnvelope(
            status=AgentStatus.success,
            agent_name="sql_agent",
            summary="SQL 완료",
            artifact_refs=[ArtifactRef(artifact_id="artifact_sql_result", type=ArtifactType.sql_result)],
            next_handoff="validation_agent",
        )


def _state():
    return empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )


def test_subagent_adapter_calls_existing_agent_contract_and_compacts_result() -> None:
    state = _state()
    adapter = SubAgentAdapter(backend_adapter=FakeAdapter(), agents={"sql_agent": StubAgent()})

    result = adapter.call("sql_agent", state)

    assert result.agent_result.agent == "sql_agent"
    assert result.agent_result.status == "success"
    assert result.agent_result.artifact_ids == ["artifact_sql_result"]
    assert result.state_updates["generated_sql"] == "SELECT 1 AS sample_value"


class ApprovalAgent:
    name = "analysis_agent"

    def run(self, state: OrchestrationState, runtime) -> AgentEnvelope:
        return AgentEnvelope(
            status=AgentStatus.success,
            agent_name="analysis_agent",
            summary="검토 필요",
            approval=ApprovalRequirement(required=True, reason="사람 검토 필요", approval_type="analysis.review"),
        )


def test_subagent_adapter_promotes_success_with_required_approval() -> None:
    adapter = SubAgentAdapter(backend_adapter=FakeAdapter(), agents={"analysis_agent": ApprovalAgent()})

    result = adapter.call("analysis_agent", _state())

    assert result.agent_result.status == "approval_required"


class BusinessFlagAgent:
    name = "eda_agent"

    def run(self, state: OrchestrationState, runtime) -> AgentEnvelope:
        return AgentEnvelope(
            status=AgentStatus.warning,
            agent_name="eda_agent",
            summary="검증 경고",
            validation=ValidationBlock(
                local_checks=[LocalCheck(name="row_count", passed=False, severity="warning", detail="행 수가 적음")],
                business_flags=[
                    BusinessFlag(code="margin", severity="warning", message="마진 급락"),
                    BusinessFlag(code="revenue", severity="error", message="매출 누락"),
                ],
            ),
        )


def test_subagent_adapter_compacts_business_flag_validation_messages() -> None:
    adapter = SubAgentAdapter(backend_adapter=FakeAdapter(), agents={"eda_agent": BusinessFlagAgent()})

    result = adapter.call("eda_agent", _state())

    assert "행 수가 적음" in result.agent_result.validation_warnings
    assert "마진 급락" in result.agent_result.validation_warnings
    assert "매출 누락" in result.agent_result.validation_errors


class LargePreviewAdapter(FakeAdapter):
    def get_artifact(self, artifact_id: str) -> FakeArtifactRecord:
        return FakeArtifactRecord(
            artifact_id=artifact_id,
            type="sql_result",
            metadata={"kind": "sql_result"},
            preview={
                "rows": [{"value": "가" * 400} for _ in range(20)],
                "columns": ["x" * 400 for _ in range(20)],
                "note": "y" * 400,
            },
        )


def test_subagent_adapter_limits_preview_summary_size() -> None:
    adapter = SubAgentAdapter(backend_adapter=LargePreviewAdapter(), agents={"sql_agent": StubAgent()})

    result = adapter.call("sql_agent", _state())

    summary = result.agent_result.artifacts[0].summary
    assert len(summary) <= 1000
    assert "x" * 241 not in summary
    assert "y" * 241 not in summary


class FailedAgent:
    name = "sql_agent"

    def run(self, state: OrchestrationState, runtime) -> AgentEnvelope:
        return AgentEnvelope(
            status=AgentStatus.failed,
            agent_name="sql_agent",
            summary="SQL 실행 실패",
            retry_hint=RetryHint(retryable=True, reason_code="SQL_TIMEOUT"),
        )


def test_subagent_adapter_preserves_failed_retry_hint_and_error() -> None:
    adapter = SubAgentAdapter(backend_adapter=FakeAdapter(), agents={"sql_agent": FailedAgent()})

    result = adapter.call("sql_agent", _state())

    assert result.agent_result.status == "failed"
    assert result.agent_result.retryable is True
    assert "SQL 실행 실패" in result.agent_result.error
    assert "SQL_TIMEOUT" in result.agent_result.error


class PlanMutationAgent:
    name = "sql_agent"

    def run(self, state: OrchestrationState, runtime) -> AgentEnvelope:
        state.planner_mode = "llm"
        state.generated_sql = "SELECT revenue FROM sales"
        assert state.plan is not None
        state.plan.generated_sql = "SELECT revenue FROM sales"
        state.plan.source_sql = "SELECT * FROM sales"
        state.plan.planner_mode = "llm"
        state.plan.route_kind = "trend"
        state.plan.goal = "월별 매출"
        return AgentEnvelope(status=AgentStatus.success, agent_name="sql_agent", summary="SQL 완료")


def test_subagent_adapter_returns_plan_metadata_state_updates() -> None:
    adapter = SubAgentAdapter(backend_adapter=FakeAdapter(), agents={"sql_agent": PlanMutationAgent()})

    result = adapter.call("sql_agent", _state())

    assert result.state_updates["planner_mode"] == "llm"
    assert result.state_updates["generated_sql"] == "SELECT revenue FROM sales"
    assert result.state_updates["analysis_plan"] == {
        "generated_sql": "SELECT revenue FROM sales",
        "source_sql": "SELECT * FROM sales",
        "planner_mode": "llm",
        "route_kind": "trend",
        "goal": "월별 매출",
    }


class InvalidStatusAgent:
    name = "sql_agent"

    def run(self, state: OrchestrationState, runtime) -> AgentEnvelope:
        return AgentEnvelope.model_construct(
            status="unknown_status",
            agent_name="sql_agent",
            summary="잘못된 상태",
            artifact_refs=[],
            validation=ValidationBlock(),
            retry_hint=RetryHint(),
            approval=ApprovalRequirement(),
            context_refs=[],
            next_handoff="",
        )


def test_subagent_adapter_rejects_invalid_status_with_agent_context() -> None:
    adapter = SubAgentAdapter(backend_adapter=FakeAdapter(), agents={"sql_agent": InvalidStatusAgent()})

    with pytest.raises(ValueError, match="sql_agent.*unknown_status"):
        adapter.call("sql_agent", _state())


def test_subagent_adapter_allows_empty_agent_mapping_without_loading_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_default_agents():
        raise AssertionError("default_agents must not be called")

    monkeypatch.setattr(tools, "default_agents", fail_default_agents)
    adapter = SubAgentAdapter(backend_adapter=FakeAdapter(), agents={})

    with pytest.raises(ValueError, match="Unknown supervisor sub-agent: sql_agent"):
        adapter.call("sql_agent", _state())

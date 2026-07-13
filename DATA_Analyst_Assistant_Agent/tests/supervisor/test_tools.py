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
    ValidationFinding,
    ValidationBlock,
)
from DATA_Analyst_Assistant_Agent.supervisor.state import empty_supervisor_state
from DATA_Analyst_Assistant_Agent.supervisor import tools
from DATA_Analyst_Assistant_Agent.supervisor.tools import AgentContractError, SubAgentAdapter
from DATA_Analyst_Assistant_Agent.supervisor.validation import validate_subagent_result


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

    def read_artifact_bytes(self, artifact_id: str) -> bytes:
        return f"bytes:{artifact_id}".encode("utf-8")


class StubAgent:
    name = "sql_agent"

    def run(self, state: OrchestrationState, runtime) -> AgentEnvelope:
        state.generated_sql = "SELECT 1 AS sample_value"
        return AgentEnvelope(
            status=AgentStatus.success,
            agent_name="sql_agent",
            summary="SQL 완료",
            artifact_refs=[ArtifactRef(artifact_id="artifact_sql_result", type=ArtifactType.sql_result)],
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

    assert result.agent_result.status == "success"
    assert result.agent_result.approval.required is True


class ChartAwareAnalysisAgent:
    name = "analysis_agent"

    def __init__(self) -> None:
        self.loaded_bytes: bytes | None = None
        self.chart_reader: Any | None = None

    def run(
        self,
        state: OrchestrationState,
        runtime,
        *,
        chart_artifact_loader=None,
        chart_reader=None,
    ) -> AgentEnvelope:
        assert chart_artifact_loader is not None
        self.loaded_bytes = chart_artifact_loader("artifact_chart")
        self.chart_reader = chart_reader
        return AgentEnvelope(
            status=AgentStatus.success,
            agent_name="analysis_agent",
            summary="analysis complete",
        )


def test_subagent_adapter_injects_chart_loader_and_reader_for_analysis_agent() -> None:
    agent = ChartAwareAnalysisAgent()
    reader = object()
    adapter = SubAgentAdapter(
        backend_adapter=FakeAdapter(),
        agents={"analysis_agent": agent},
        chart_reader=reader,
    )

    result = adapter.call("analysis_agent", _state())

    assert result.agent_result.status == "success"
    assert agent.loaded_bytes == b"bytes:artifact_chart"
    assert agent.chart_reader is reader


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
            retry_hint=RetryHint(
                retryable=True,
                reason_code="SQL_TIMEOUT",
                details={"failure_reason": "database did not respond"},
            ),
            error="원본 SQL timeout 오류",
        )


def test_subagent_adapter_preserves_failed_retry_hint_and_error() -> None:
    adapter = SubAgentAdapter(backend_adapter=FakeAdapter(), agents={"sql_agent": FailedAgent()})

    result = adapter.call("sql_agent", _state())

    assert result.agent_result.status == "failed"
    assert result.agent_result.retryable is True
    assert result.agent_result.error == "원본 SQL timeout 오류"
    assert result.agent_result.retry_hint.reason_code == "SQL_TIMEOUT"
    assert result.agent_result.retry_hint.details["failure_reason"] == "database did not respond"


class StructuredFindingAgent:
    name = "sql_agent"

    def run(self, state: OrchestrationState, runtime) -> AgentEnvelope:
        return AgentEnvelope(
            status=AgentStatus.success,
            agent_name="sql_agent",
            summary="재시도가 필요한 SQL 결과",
            validation=ValidationBlock(
                findings=[
                    ValidationFinding(
                        code="invalid_join_plan",
                        source="sql_langgraph",
                        severity="warning",
                        disposition="retry_required",
                        message="조인 계획을 다시 생성해야 합니다.",
                        retryable=True,
                        suggested_action="fix_sql",
                    )
                ]
            ),
            retry_hint=RetryHint(
                retryable=True,
                suggested_action="fix_sql",
                reason_code="invalid_join_plan",
                details={"join": "orders-customers"},
            ),
            approval=ApprovalRequirement(
                required=True,
                reason="실행 승인 필요",
                approval_type="sql.execute",
            ),
        )


def test_subagent_adapter_preserves_structured_findings_retry_hint_and_approval() -> None:
    adapter = SubAgentAdapter(
        backend_adapter=FakeAdapter(),
        agents={"sql_agent": StructuredFindingAgent()},
    )

    result = adapter.call("sql_agent", _state()).agent_result

    assert result.status == "success"
    assert result.findings[0].code == "invalid_join_plan"
    assert result.findings[0].disposition == "retry_required"
    assert result.retry_hint.reason_code == "invalid_join_plan"
    assert result.retry_hint.details == {"join": "orders-customers"}
    assert result.approval.required is True


class FallbackAgent:
    name = "analysis_agent"

    def run(self, state: OrchestrationState, runtime) -> AgentEnvelope:
        return AgentEnvelope(
            status=AgentStatus.success,
            agent_name="analysis_agent",
            summary="fallback 분석 결과",
            fallback_used=True,
            retry_hint=RetryHint(retryable=True),
        )


def test_subagent_adapter_propagates_fallback_contract_to_validation() -> None:
    state = _state()
    adapter = SubAgentAdapter(backend_adapter=FakeAdapter(), agents={"analysis_agent": FallbackAgent()})

    result = adapter.call("analysis_agent", state)
    decision = validate_subagent_result(state, result.agent_result)

    assert result.agent_result.fallback_used is True
    assert decision.valid is False
    assert decision.next_action == "call_analysis_agent"


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
    state = _state()
    state["analysis_plan"] = {"goal": "월별 매출 추이 분석"}

    result = adapter.call("sql_agent", state)

    assert result.state_updates["planner_mode"] == "llm"
    assert result.state_updates["generated_sql"] == "SELECT revenue FROM sales"
    assert result.state_updates["analysis_plan"] == {
        "generated_sql": "SELECT revenue FROM sales",
        "source_sql": "SELECT * FROM sales",
        "planner_mode": "llm",
        "route_kind": "trend",
        "goal": "월별 매출",
        "target_table": None,
        "source_tables": [],
        "business_grain": None,
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


class RaisingAgent:
    name = "eda_agent"

    def run(self, state: OrchestrationState, runtime) -> AgentEnvelope:
        state.generated_sql = "SELECT partial_result FROM unsafe_state"
        raise RuntimeError("missing upstream SQL artifact")


def test_subagent_adapter_converts_agent_exception_to_failed_contract() -> None:
    adapter = SubAgentAdapter(backend_adapter=FakeAdapter(), agents={"eda_agent": RaisingAgent()})

    result = adapter.call("eda_agent", _state())

    assert result.agent_result.agent == "eda_agent"
    assert result.agent_result.status == "failed"
    assert result.agent_result.retryable is True
    assert result.agent_result.retry_hint.retryable is True
    assert result.agent_result.retry_hint.reason_code == "agent_execution_exception"
    assert result.agent_result.retry_hint.details == {
        "failure_reason": "missing upstream SQL artifact",
        "exception_type": "RuntimeError",
    }
    assert "missing upstream SQL artifact" in result.agent_result.error
    assert result.state_updates == {}


class MismatchedNameAgent:
    name = "sql_agent"

    def run(self, state: OrchestrationState, runtime) -> AgentEnvelope:
        return AgentEnvelope(status=AgentStatus.success, agent_name="eda_agent", summary="wrong envelope")


def test_subagent_adapter_rejects_mismatched_agent_name() -> None:
    adapter = SubAgentAdapter(backend_adapter=FakeAdapter(), agents={"sql_agent": MismatchedNameAgent()})

    with pytest.raises(AgentContractError, match="sql_agent.*eda_agent"):
        adapter.call("sql_agent", _state())


class DetailedApprovalAgent:
    name = "analysis_agent"

    def run(self, state: OrchestrationState, runtime) -> AgentEnvelope:
        return AgentEnvelope(
            status=AgentStatus.success,
            agent_name="analysis_agent",
            summary="needs review",
            approval=ApprovalRequirement(
                required=True,
                reason="review causal assumptions",
                approval_type="analysis.review",
            ),
        )


@pytest.mark.xfail(reason="AgentCompactResult should preserve approval reason/type, not only status.")
def test_subagent_adapter_preserves_approval_details_in_compact_result() -> None:
    adapter = SubAgentAdapter(backend_adapter=FakeAdapter(), agents={"analysis_agent": DetailedApprovalAgent()})

    result = adapter.call("analysis_agent", _state())

    assert result.agent_result.approval_reason == "review causal assumptions"
    assert result.agent_result.approval_type == "analysis.review"


class IntegrityRefAgent:
    name = "sql_agent"

    def run(self, state: OrchestrationState, runtime) -> AgentEnvelope:
        return AgentEnvelope(
            status=AgentStatus.success,
            agent_name="sql_agent",
            summary="SQL complete with validation artifact",
            validation=ValidationBlock(
                integrity_refs=[
                    ArtifactRef(artifact_id="artifact_integrity_summary", type=ArtifactType.file),
                ],
            ),
        )


@pytest.mark.xfail(reason="AgentCompactResult should preserve ValidationBlock.integrity_refs for supervisor output.")
def test_subagent_adapter_preserves_integrity_refs_in_compact_result() -> None:
    adapter = SubAgentAdapter(backend_adapter=FakeAdapter(), agents={"sql_agent": IntegrityRefAgent()})

    result = adapter.call("sql_agent", _state())

    assert result.agent_result.integrity_ref_ids == ["artifact_integrity_summary"]

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_agent_backend.models.artifacts import ArtifactRef, ArtifactType

from DATA_Analyst_Assistant_Agent.shared.contracts import AgentEnvelope, AgentStatus, OrchestrationState
from DATA_Analyst_Assistant_Agent.supervisor.state import empty_supervisor_state
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


def test_subagent_adapter_calls_existing_agent_contract_and_compacts_result() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )
    adapter = SubAgentAdapter(backend_adapter=FakeAdapter(), agents={"sql_agent": StubAgent()})

    result = adapter.call("sql_agent", state)

    assert result.agent_result.agent == "sql_agent"
    assert result.agent_result.status == "success"
    assert result.agent_result.artifact_ids == ["artifact_sql_result"]
    assert result.state_updates["generated_sql"] == "SELECT 1 AS sample_value"

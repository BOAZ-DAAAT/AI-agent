from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.supervisor.insight.agent import InsightGenerator
from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    SupervisorState,
    to_orchestration_state,
)
from DATA_Analyst_Assistant_Agent.supervisor.tools import compact_agent_envelope


class SupervisorInsightGenerator:
    def __init__(self, backend_adapter: BackendAdapter) -> None:
        self.backend_adapter = backend_adapter
        self.runtime = AgentRuntime(adapter=backend_adapter)
        self._generator = InsightGenerator()

    def generate(self, state: SupervisorState) -> AgentCompactResult:
        orchestration_state = to_orchestration_state(state)
        envelope = self._generator.run(orchestration_state, self.runtime)
        return compact_agent_envelope(envelope, self.backend_adapter)

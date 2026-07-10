from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.agents.report.service import generate_report_envelope
from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    SupervisorState,
    to_orchestration_state,
)
from DATA_Analyst_Assistant_Agent.supervisor.tools import compact_agent_envelope


class SupervisorReportGenerator:
    def __init__(self, backend_adapter: BackendAdapter) -> None:
        self.backend_adapter = backend_adapter
        self.runtime = AgentRuntime(adapter=backend_adapter)

    def generate(self, state: SupervisorState) -> AgentCompactResult:
        orchestration_state = to_orchestration_state(state)
        envelope = generate_report_envelope(
            orchestration_state,
            self.runtime,
            node_name="generate_report",
        )
        return compact_agent_envelope(envelope, self.backend_adapter)

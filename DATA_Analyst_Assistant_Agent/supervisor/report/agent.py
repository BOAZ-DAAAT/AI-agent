from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.supervisor.report.service import generate_report_envelope
from DATA_Analyst_Assistant_Agent.shared.contracts import AgentEnvelope, OrchestrationState


class ReportGenerator:
    name = "report_agent"

    def run(self, state: OrchestrationState, runtime: AgentRuntime) -> AgentEnvelope:
        return generate_report_envelope(state, runtime, node_name=self.name)

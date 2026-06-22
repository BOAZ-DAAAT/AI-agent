from __future__ import annotations

from dataclasses import dataclass

from data_agent_backend.models.contexts import PolicyContext

from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState


@dataclass
class AgentRuntime:
    adapter: BackendAdapter

    def context(self, state: OrchestrationState, *, node_name: str, tool_name: str) -> PolicyContext:
        return PolicyContext(
            run_id=state.run_id,
            thread_id=state.thread_id,
            node_name=node_name,
            tool_name=tool_name,
        )

from __future__ import annotations

from importlib import import_module

__all__ = [
    "AgentEnvelope",
    "AgentStatus",
    "AnalysisAgent",
    "BackendAdapter",
    "EDAAgent",
    "OrchestrationState",
    "ReportAgent",
    "SQLAgent",
    "SQLAgentSupervisor",
    "SupervisorAgent",
    "SupervisorInterruptPayload",
    "SupervisorRunResult",
    "SupervisorTerminalState",
    "build_graph",
]

_EXPORT_MAP = {
    "AgentEnvelope": ("DATA_Analyst_Assistant_Agent.shared.contracts", "AgentEnvelope"),
    "AgentStatus": ("DATA_Analyst_Assistant_Agent.shared.contracts", "AgentStatus"),
    "AnalysisAgent": ("DATA_Analyst_Assistant_Agent.agents", "AnalysisAgent"),
    "BackendAdapter": ("DATA_Analyst_Assistant_Agent.shared.backend_adapter", "BackendAdapter"),
    "EDAAgent": ("DATA_Analyst_Assistant_Agent.agents", "EDAAgent"),
    "OrchestrationState": ("DATA_Analyst_Assistant_Agent.shared.contracts", "OrchestrationState"),
    "ReportAgent": ("DATA_Analyst_Assistant_Agent.agents", "ReportAgent"),
    "SQLAgent": ("DATA_Analyst_Assistant_Agent.agents", "SQLAgent"),
    "SQLAgentSupervisor": ("DATA_Analyst_Assistant_Agent.supervisor", "SQLAgentSupervisor"),
    "SupervisorAgent": ("DATA_Analyst_Assistant_Agent.supervisor", "SupervisorAgent"),
    "SupervisorInterruptPayload": ("DATA_Analyst_Assistant_Agent.shared.contracts", "SupervisorInterruptPayload"),
    "SupervisorRunResult": ("DATA_Analyst_Assistant_Agent.shared.contracts", "SupervisorRunResult"),
    "SupervisorTerminalState": ("DATA_Analyst_Assistant_Agent.shared.contracts", "SupervisorTerminalState"),
    "build_graph": ("DATA_Analyst_Assistant_Agent.supervisor.graph", "build_graph"),
}

for _deleted_export in ("CentralValidationAgent", "VisualizationAgent"):
    globals().pop(_deleted_export, None)


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(name)
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

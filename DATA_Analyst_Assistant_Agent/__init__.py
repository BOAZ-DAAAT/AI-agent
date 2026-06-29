from __future__ import annotations

from importlib import import_module

__all__ = [
    "AgentEnvelope",
    "AgentStatus",
    "AnalysisAgent",
    "BackendAdapter",
    "CentralValidationAgent",
    "EDAAgent",
    "OrchestrationState",
    "ReportAgent",
    "SQLAgent",
    "SQLAgentSupervisor",
    "SupervisorTerminalState",
    "VisualizationAgent",
    "build_graph",
]

_EXPORT_MAP = {
    "AgentEnvelope": ("DATA_Analyst_Assistant_Agent.shared.contracts", "AgentEnvelope"),
    "AgentStatus": ("DATA_Analyst_Assistant_Agent.shared.contracts", "AgentStatus"),
    "AnalysisAgent": ("DATA_Analyst_Assistant_Agent.agents", "AnalysisAgent"),
    "BackendAdapter": ("DATA_Analyst_Assistant_Agent.shared.backend_adapter", "BackendAdapter"),
    "CentralValidationAgent": ("DATA_Analyst_Assistant_Agent.agents", "CentralValidationAgent"),
    "EDAAgent": ("DATA_Analyst_Assistant_Agent.agents", "EDAAgent"),
    "OrchestrationState": ("DATA_Analyst_Assistant_Agent.shared.contracts", "OrchestrationState"),
    "ReportAgent": ("DATA_Analyst_Assistant_Agent.agents", "ReportAgent"),
    "SQLAgent": ("DATA_Analyst_Assistant_Agent.agents", "SQLAgent"),
    "SQLAgentSupervisor": ("DATA_Analyst_Assistant_Agent.supervisor", "SQLAgentSupervisor"),
    "SupervisorTerminalState": ("DATA_Analyst_Assistant_Agent.shared.contracts", "SupervisorTerminalState"),
    "VisualizationAgent": ("DATA_Analyst_Assistant_Agent.agents", "VisualizationAgent"),
    "build_graph": ("DATA_Analyst_Assistant_Agent.supervisor.graph", "build_graph"),
}


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(name)
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

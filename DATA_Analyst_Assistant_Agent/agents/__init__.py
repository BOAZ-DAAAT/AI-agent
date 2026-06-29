from __future__ import annotations

from importlib import import_module

__all__ = [
    "AgentRuntime",
    "AnalysisAgent",
    "CentralValidationAgent",
    "EDAAgent",
    "ReportAgent",
    "SQLAgent",
    "VisualizationAgent",
]

_EXPORT_MAP = {
    "AgentRuntime": ("DATA_Analyst_Assistant_Agent.agents.common", "AgentRuntime"),
    "AnalysisAgent": ("DATA_Analyst_Assistant_Agent.agents.analysis.agent", "AnalysisAgent"),
    "CentralValidationAgent": ("DATA_Analyst_Assistant_Agent.agents.validation.agent", "CentralValidationAgent"),
    "EDAAgent": ("DATA_Analyst_Assistant_Agent.agents.eda.agent", "EDAAgent"),
    "ReportAgent": ("DATA_Analyst_Assistant_Agent.agents.report.agent", "ReportAgent"),
    "SQLAgent": ("DATA_Analyst_Assistant_Agent.agents.sql.agent", "SQLAgent"),
    "VisualizationAgent": ("DATA_Analyst_Assistant_Agent.agents.visualization.agent", "VisualizationAgent"),
}


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(name)
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

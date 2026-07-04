from __future__ import annotations

from importlib import import_module

__all__ = ["SupervisorAgent", "SQLAgentSupervisor", "build_graph"]

_EXPORT_MAP = {
    "SupervisorAgent": ("DATA_Analyst_Assistant_Agent.supervisor.agent", "SupervisorAgent"),
    "SQLAgentSupervisor": ("DATA_Analyst_Assistant_Agent.supervisor.agent", "SQLAgentSupervisor"),
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

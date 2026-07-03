from __future__ import annotations

import importlib


DELETED_AGENT_EXPORTS = ("CentralValidationAgent", "VisualizationAgent")


def test_package_exports_new_supervisor_aliases() -> None:
    import DATA_Analyst_Assistant_Agent as daaa

    assert daaa.SupervisorAgent is daaa.SQLAgentSupervisor
    assert callable(daaa.build_graph)


def test_packages_no_longer_export_deleted_agents() -> None:
    import DATA_Analyst_Assistant_Agent as daaa
    import DATA_Analyst_Assistant_Agent.agents as agents

    for name in DELETED_AGENT_EXPORTS:
        assert name not in daaa.__all__
        assert name not in agents.__all__
        assert not hasattr(daaa, name)
        assert not hasattr(agents, name)


def test_reload_clears_stale_deleted_agent_globals() -> None:
    import DATA_Analyst_Assistant_Agent as daaa
    import DATA_Analyst_Assistant_Agent.agents as agents

    for name in DELETED_AGENT_EXPORTS:
        setattr(daaa, name, object())
        setattr(agents, name, object())

    importlib.reload(daaa)
    importlib.reload(agents)

    for name in DELETED_AGENT_EXPORTS:
        assert not hasattr(daaa, name)
        assert not hasattr(agents, name)

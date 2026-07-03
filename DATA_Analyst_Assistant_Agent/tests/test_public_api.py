from __future__ import annotations


def test_package_exports_new_supervisor_aliases() -> None:
    import DATA_Analyst_Assistant_Agent as daaa

    assert daaa.SupervisorAgent is daaa.SQLAgentSupervisor
    assert callable(daaa.build_graph)


def test_agents_package_no_longer_exports_deleted_agents() -> None:
    import DATA_Analyst_Assistant_Agent.agents as agents

    assert "CentralValidationAgent" not in agents.__all__
    assert "VisualizationAgent" not in agents.__all__

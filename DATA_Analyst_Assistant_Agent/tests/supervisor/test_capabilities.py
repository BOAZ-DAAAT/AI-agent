from __future__ import annotations

import json

from DATA_Analyst_Assistant_Agent.supervisor.capabilities import DEFAULT_AGENT_CAPABILITIES
from DATA_Analyst_Assistant_Agent.supervisor.graph import ACTION_TO_AGENT, SUBAGENT_ACTION_TO_AGENT


def test_default_capabilities_include_all_supervisor_agents() -> None:
    capabilities_by_agent = {capability.agent: capability for capability in DEFAULT_AGENT_CAPABILITIES}

    assert set(capabilities_by_agent) == {
        "sql_agent",
        "eda_agent",
        "analysis_agent",
        "report_agent",
    }


def test_default_capability_actions_match_action_to_agent_mapping() -> None:
    actions_by_agent = {agent: action for action, agent in ACTION_TO_AGENT.items()}

    for capability in DEFAULT_AGENT_CAPABILITIES:
        assert capability.action == actions_by_agent[capability.agent]


def test_subagent_action_mapping_excludes_report_agent() -> None:
    assert SUBAGENT_ACTION_TO_AGENT == {
        "call_sql_agent": "sql_agent",
        "call_eda_agent": "eda_agent",
        "call_analysis_agent": "analysis_agent",
    }


def test_default_capabilities_are_json_serializable() -> None:
    payload = [capability.model_dump(mode="json") for capability in DEFAULT_AGENT_CAPABILITIES]

    json.dumps(payload, ensure_ascii=False)


def test_default_capabilities_capture_or_artifact_preconditions() -> None:
    capabilities_by_agent = {capability.agent: capability for capability in DEFAULT_AGENT_CAPABILITIES}

    assert capabilities_by_agent["analysis_agent"].requires_any_artifacts_from == [
        "sql_agent",
        "eda_agent",
    ]
    assert capabilities_by_agent["report_agent"].requires_any_artifacts_from == [
        "sql_agent",
        "eda_agent",
        "analysis_agent",
    ]

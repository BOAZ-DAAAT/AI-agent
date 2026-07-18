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
        "insight",
    }


def test_default_capability_actions_match_action_to_agent_mapping() -> None:
    actions_by_agent = {agent: action for action, agent in ACTION_TO_AGENT.items()}

    for capability in DEFAULT_AGENT_CAPABILITIES:
        assert capability.action == actions_by_agent[capability.agent]


def test_subagent_action_mapping_excludes_insight() -> None:
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
    assert capabilities_by_agent["insight"].requires_any_artifacts_from == [
        "sql_agent",
        "eda_agent",
        "analysis_agent",
    ]


def test_default_capabilities_declare_required_output_evidence_contracts() -> None:
    capabilities = {item.agent: item for item in DEFAULT_AGENT_CAPABILITIES}

    assert [(item.type, item.kind) for item in capabilities["sql_agent"].output_evidence] == [
        ("sql_query", "generated_sql"),
        ("sql_result", "sql_result"),
    ]
    assert [(item.type, item.kind) for item in capabilities["eda_agent"].output_evidence] == [
        ("data_profile", "eda_summary")
    ]
    assert [(item.type, item.kind) for item in capabilities["analysis_agent"].output_evidence] == [
        ("file", "analysis_result")
    ]
    assert [(item.type, item.kind) for item in capabilities["insight"].output_evidence] == [
        ("file", "insight_payload")
    ]
    assert capabilities["analysis_agent"].input_evidence_mode == "any"
    assert capabilities["insight"].input_evidence_mode == "any"


def test_default_capabilities_describe_sql_as_data_preparation_only() -> None:
    capabilities = {item.agent: item for item in DEFAULT_AGENT_CAPABILITIES}
    sql = capabilities["sql_agent"]

    assert "분석용 SQL 결과" in sql.description
    assert "데이터셋" in sql.when_to_use
    avoid_text = " ".join(sql.avoid_when)
    for keyword in ["분포", "결측", "이상치", "품질", "가설 후보", "분석 방향"]:
        assert keyword in avoid_text


def test_default_capabilities_describe_eda_as_exploratory_discovery() -> None:
    capabilities = {item.agent: item for item in DEFAULT_AGENT_CAPABILITIES}
    eda = capabilities["eda_agent"]

    eda_text = " ".join([eda.description, eda.when_to_use])
    for keyword in ["탐색", "발견", "가설 후보", "분석 방향"]:
        assert keyword in eda_text
    assert eda.produces_artifacts == ["data_profile", "quality_summary", "eda_summary"]
    assert [(item.type, item.kind) for item in eda.output_evidence] == [
        ("data_profile", "eda_summary")
    ]


def test_default_capabilities_describe_analysis_as_analysis_executor() -> None:
    capabilities = {item.agent: item for item in DEFAULT_AGENT_CAPABILITIES}
    analysis = capabilities["analysis_agent"]

    analysis_text = " ".join([analysis.description, analysis.when_to_use])
    for keyword in ["분석을 설계", "수행", "통계 검정", "모델링", "비즈니스 인사이트"]:
        assert keyword in analysis_text
    assert "EDA 결과가 없을 때" in " ".join(analysis.avoid_when)

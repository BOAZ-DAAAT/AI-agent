from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from DATA_Analyst_Assistant_Agent.supervisor.state import AgentName, NextAction


class AgentCapability(BaseModel):
    agent: AgentName
    action: NextAction
    description: str
    when_to_use: str
    requires_artifacts_from: list[AgentName] = Field(default_factory=list)
    requires_any_artifacts_from: list[AgentName] = Field(default_factory=list)
    produces_artifacts: list[str] = Field(default_factory=list)
    avoid_when: list[str] = Field(default_factory=list)
    input_evidence: list["EvidenceRequirement"] = Field(default_factory=list)
    input_evidence_mode: Literal["all", "any"] = "all"
    output_evidence: list["EvidenceRequirement"] = Field(default_factory=list)


class EvidenceRequirement(BaseModel):
    type: str
    kind: str
    non_empty: bool = False


DEFAULT_AGENT_CAPABILITIES: list[AgentCapability] = [
    AgentCapability(
        agent="sql_agent",
        action="call_sql_agent",
        description="조회·조인·필터링·집계로 분석용 SQL 결과를 생성합니다.",
        when_to_use="원천 데이터에서 필요한 데이터셋, 집계표, 기간/그룹별 결과를 만들 때 사용합니다.",
        produces_artifacts=["sql_query", "sql_result_csv"],
        output_evidence=[
            EvidenceRequirement(type="sql_query", kind="generated_sql", non_empty=True),
            EvidenceRequirement(type="sql_result", kind="sql_result"),
        ],
        avoid_when=[
            "이미 충분한 SQL 결과 artifact가 있고 재조회가 필요하지 않을 때",
            "SQL 결과의 분포, 결측, 이상치, 품질, 기본 패턴 해석이 필요할 때",
            "가설 후보, 분석 방향, 비즈니스 인사이트 도출이 필요할 때",
        ],
    ),
    AgentCapability(
        agent="eda_agent",
        action="call_eda_agent",
        description="SQL 결과를 탐색해 분포·결측·이상치·품질·패턴을 발견하고 가설 후보와 분석 방향을 제안합니다.",
        when_to_use="SQL 결과가 준비된 뒤 데이터 특성, 숨은 패턴, 이상 징후, 가설 후보, 분석 방향을 탐색해야 할 때 사용합니다.",
        requires_artifacts_from=["sql_agent"],
        produces_artifacts=["data_profile", "quality_summary", "eda_summary"],
        input_evidence=[EvidenceRequirement(type="sql_result", kind="sql_result")],
        output_evidence=[EvidenceRequirement(type="data_profile", kind="eda_summary")],
        avoid_when=["SQL 결과 artifact가 없을 때"],
    ),
    AgentCapability(
        agent="analysis_agent",
        action="call_analysis_agent",
        description="SQL/EDA 근거로 분석을 설계·수행하고 통계 검정·모델링·원인 해석·비즈니스 인사이트를 도출합니다.",
        when_to_use="사용자 질문에 답하기 위해 비교, 추세 해석, 원인 분석, 통계 검정, 모델링, 세그먼트 해석 등 분석 수행이 필요할 때 사용합니다.",
        requires_any_artifacts_from=["sql_agent", "eda_agent"],
        produces_artifacts=["analysis_result", "modeling_summary", "business_insight"],
        input_evidence=[
            EvidenceRequirement(type="sql_result", kind="sql_result"),
            EvidenceRequirement(type="data_profile", kind="eda_summary"),
        ],
        input_evidence_mode="any",
        output_evidence=[EvidenceRequirement(type="file", kind="analysis_result")],
        avoid_when=[
            "근거가 되는 SQL 또는 EDA artifact가 없을 때",
            "분포·결측·이상치·품질 확인이 먼저 필요한데 EDA 결과가 없을 때",
        ],
    ),
    AgentCapability(
        agent="report_agent",
        action="call_report_agent",
        description="기존 근거 artifact를 통합해 최종 Markdown report를 생성합니다.",
        when_to_use="분석 근거가 준비되어 사용자에게 전달할 최종 리포트가 필요할 때 사용합니다.",
        requires_any_artifacts_from=["sql_agent", "eda_agent", "analysis_agent"],
        produces_artifacts=["markdown_report"],
        input_evidence=[
            EvidenceRequirement(type="sql_result", kind="sql_result"),
            EvidenceRequirement(type="data_profile", kind="eda_summary"),
            EvidenceRequirement(type="file", kind="analysis_result"),
        ],
        input_evidence_mode="any",
        output_evidence=[EvidenceRequirement(type="report", kind="final_report", non_empty=True)],
        avoid_when=["리포트에 포함할 근거 artifact가 없을 때"],
    ),
]


def agent_capabilities_context() -> list[dict[str, object]]:
    return [capability.model_dump(mode="json") for capability in DEFAULT_AGENT_CAPABILITIES]

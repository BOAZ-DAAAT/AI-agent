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


class DisplayCapability(BaseModel):
    name: Literal["summary_agent", "report_agent"]
    description: str
    when_to_use: str
    produces_artifacts: list[str] = Field(default_factory=list)
    avoid_when: list[str] = Field(default_factory=list)


DEFAULT_AGENT_CAPABILITIES: list[AgentCapability] = [
    AgentCapability(
        agent="sql_agent",
        action="call_sql_agent",
        description=(
            "사용자 질문에 필요한 원천 데이터와 재사용 가능한 데이터마트를 만듭니다. "
            "조인, 필터, 집계, 공통 grain 정리, 원자 변수와 구조적 파생변수 생성을 담당합니다."
        ),
        when_to_use=(
            "계산식이 데이터 구조와 명확한 산술 규칙으로 정해지는 변수가 필요할 때 사용합니다. "
            "날짜 차이, 기간 버킷, 월/주/요일, 주문 수, 금액 합계, 평균, 비율, 중복 제거 후 개수처럼 "
            "별도의 분석 판단 없이 정의할 수 있는 변수는 SQL에서 만들 수 있습니다. "
            "기준값, 등급, 세그먼트, 라벨, 상태 구분처럼 데이터 분포나 결과 변수와의 관계를 보고 "
            "의미를 정해야 하는 변수는 임의로 만들지 않고, 판단에 필요한 측정값을 남깁니다."
        ),
        produces_artifacts=["sql_query", "sql_result_csv"],
        output_evidence=[
            EvidenceRequirement(type="sql_query", kind="generated_sql", non_empty=True),
            EvidenceRequirement(type="sql_result", kind="sql_result"),
        ],
        avoid_when=[
            "이미 충분한 SQL 결과 artifact가 있고 재조회가 필요하지 않을 때",
            "SQL 결과의 분포, 결측, 이상치, 품질, 기본 패턴 해석이 필요할 때",
            "기준값, 등급, 세그먼트, 라벨, 상태 구분의 의미 정의가 필요할 때",
        ],
    ),
    AgentCapability(
        agent="eda_agent",
        action="call_eda_agent",
        description=(
            "SQL 결과를 읽어 데이터의 상태와 관찰 가능한 신호를 정리합니다. "
            "분포, 결측, 이상치, 기본 통계, 그룹별 요약, 차트 패턴, 단순 관계와 추세를 확인하고, "
            "이를 토대로 analysis_agent가 검토할 탐색 근거와 후보 가설을 만듭니다. "
            "파생 지표 계산, 통계 검정, 회귀/모델링, 가설 채택·기각은 수행하지 않습니다."
        ),
        when_to_use=(
            "SQL 결과가 준비된 뒤 데이터 특성, 관찰 가능한 신호, 이상 징후, 후보 가설, "
            "분석 방향을 탐색해야 할 때 사용합니다. EDA 산출물은 최종 결론이 아니라 분석 판단의 재료입니다. "
            "관찰된 신호가 사용자 질문의 결론으로 사용될 수 있다면 수치, 시각적 근거, 한계, 후보 가설을 "
            "analysis_agent가 이어받을 수 있게 남깁니다. 질문의 핵심 답변에 p-value, 효과크기, 회귀, ANOVA, "
            "조건부 파생 지표 계산이 필요하면 EDA 이후 analysis_agent를 고려합니다."
        ),
        requires_artifacts_from=["sql_agent"],
        produces_artifacts=["data_profile", "quality_summary", "eda_summary"],
        input_evidence=[EvidenceRequirement(type="sql_result", kind="sql_result")],
        output_evidence=[EvidenceRequirement(type="data_profile", kind="eda_summary")],
        avoid_when=["SQL 결과 artifact가 없을 때"],
    ),
    AgentCapability(
        agent="analysis_agent",
        action="call_analysis_agent",
        description=(
            "SQL과 EDA가 만든 데이터 근거를 바탕으로, 관찰된 신호가 사용자 질문에 대한 답으로 "
            "얼마나 타당한지 검토하고 분석 결론의 강도와 한계를 정합니다. "
            "질문 맞춤 파생 지표 계산과 통계 분석 실행은 이 에이전트의 책임입니다."
        ),
        when_to_use=(
            "관찰된 관계, 추세, 차이, 이상치, 후보 가설을 그대로 결론으로 쓰지 않고, "
            "표본 크기, 집계 단위, 효과 크기, 민감도, 대안 설명, 데이터 한계를 함께 검토해야 할 때 사용합니다. "
            "EDA의 insight_result, hypotheses, final_summary는 후보 신호로만 참고하고, 최종 계수, p-value, "
            "검정 결과, 효과크기, 해석은 analysis_agent가 재계산한 evidence만 사용합니다. "
            "데이터에 직접 존재하지 않는 개념을 기준값, 등급, 세그먼트, 라벨, 상태 구분 등으로 "
            "분석 안에서 정의해야 할 수 있습니다. 먼저 데이터 분포, 결과 변수와의 관계, 표본 수, 민감도 등을 확인해 "
            "방어 가능한 기준이나 분석 방법을 찾고, 그 정의나 방법 선택이 결과 해석에 실질적인 영향을 주며 "
            "사용자 확인이 분석 품질에 도움이 될 때만 review_request를 만듭니다. 하나의 추천안을 확인받는 방식이 "
            "나을 수도 있고, 타당한 방법들이 서로 다른 해석상의 트레이드오프를 만들면 옵션 선택 방식이 나을 수도 있습니다."
        ),
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
        agent="insight",
        action="call_insight",
        description=(
            "SQL, EDA, analysis_agent가 만든 검증된 근거를 모아 사용자 질문에 대한 핵심 답변과 시사점을 만듭니다."
        ),
        when_to_use=(
            "분석 근거가 준비된 뒤, 사용자 질문 전체에 대한 핵심 결론, 근거, 한계, 주의사항을 간결하게 "
            "종합해야 할 때 자동으로 실행됩니다. 새로운 분석 기준, 통계 판단, 조작적 정의를 만들지 않고, "
            "analysis_agent 결과가 있으면 그 분석 판단을 우선 근거로 사용합니다. analysis_agent 없이 SQL/EDA 근거만 "
            "있는 경우에는 관찰 사실 위주로 답하고 강한 해석을 새로 만들지 않습니다."
        ),
        requires_any_artifacts_from=["sql_agent", "eda_agent", "analysis_agent"],
        produces_artifacts=["insight_payload"],
        input_evidence=[
            EvidenceRequirement(type="sql_result", kind="sql_result"),
            EvidenceRequirement(type="data_profile", kind="eda_summary"),
            EvidenceRequirement(type="file", kind="analysis_result"),
        ],
        input_evidence_mode="any",
        output_evidence=[EvidenceRequirement(type="file", kind="insight_payload", non_empty=True)],
        avoid_when=["인사이트에 포함할 근거 artifact가 없을 때"],
    ),
]

DEFAULT_DISPLAY_CAPABILITIES: list[DisplayCapability] = [
    DisplayCapability(
        name="summary_agent",
        description=(
            "각 에이전트가 만든 아티팩트를 읽어 UI에 표시할 단계별 요약 산출물을 생성합니다."
        ),
        when_to_use=(
            "SQL, EDA, analysis, insight 결과를 사용자에게 이해 가능한 패널 형태로 정리해야 할 때 사용합니다. "
            "제목, 배경, 확인 항목, 주요 발견, 관련 차트, 결론, 핵심 문장을 만들되 새로운 분석, 추가 계산, "
            "조작적 정의, 통계 판단은 만들지 않습니다. 근거 아티팩트에 있는 내용만 사용하고 숫자는 검증 가능한 값만 인용합니다."
        ),
        produces_artifacts=["node_summary"],
    ),
    DisplayCapability(
        name="report_agent",
        description=(
            "사용자 질문, 실행 경로, 에이전트별 산출물, 아티팩트, 검증 결과를 읽어 최종 보고서 산출물을 생성합니다."
        ),
        when_to_use=(
            "최종 보고서의 구조, 섹션, 요약 문장, 근거 설명, 표, 차트 배치, 방법 설명, 한계 정리가 필요할 때 사용합니다. "
            "이미 생성된 SQL, EDA, analysis, insight 근거를 바탕으로 사용자에게 보여줄 표를 만들 수 있지만, "
            "새로운 분석 기준, 통계 판단, 조작적 정의, 검정 결과를 임의로 만들지 않습니다."
        ),
        produces_artifacts=["report"],
    ),
]


def agent_capabilities_context() -> list[dict[str, object]]:
    return [capability.model_dump(mode="json") for capability in DEFAULT_AGENT_CAPABILITIES]


def display_capabilities_context() -> list[dict[str, object]]:
    return [capability.model_dump(mode="json") for capability in DEFAULT_DISPLAY_CAPABILITIES]

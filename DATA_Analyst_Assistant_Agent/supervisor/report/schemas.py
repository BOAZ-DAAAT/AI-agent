"""리포트 결과 계약 — SQL→EDA→분석→인사이트 경로 전체를 하나의 완성된 글로.

summary의 "노드 하나" 계약(NodeSummaryResult)과 달리, 이건 "경로 전체(존재하는 단계만)"를
다룬다. 단계별로 나열하는 게 목적이 아니라 하나의 흐름 있는 보고서로 종합하는 게 목적이라
필드도 그에 맞춰 다르게 짰다(background_and_question/methodology_narrative처럼 여러 단계를
한 필드에 녹인다).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FindingSection(BaseModel):
    """key_findings 리스트의 항목 하나 — 소제목 + 본문 + (있으면) 관련 차트들."""

    heading: str
    body: str
    source_label: str | None = None
    chart_artifact_ids: list[str] = Field(default_factory=list)


class ReportResult(BaseModel):
    """경로 전체(SQL/EDA/분석/인사이트 중 실제로 존재하는 단계)를 종합한 리포트. fallback 경로에서도 전부 채워진다."""

    title: str
    executive_summary: str                                # 결론부터, 3~5문장
    background_and_question: str                           # 사용자가 뭘 궁금해했는지 + 왜 이 여정이 필요했는지
    methodology_narrative: str                              # SQL+EDA를 하나로 묶은 "이렇게 준비·검증했다" 서술
    code_used: str = ""                                    # 근거에서 그대로 가져온 코드/SQL(LLM이 쓰지 않음)
    key_findings: list[FindingSection] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    conclusion_and_recommendations: str                     # 결론 + 실행 제안(인사이트의 action_plan 흡수)
    key_finding: str                                        # 전체를 압축한 한 문장(미리보기용)
    included_stages: list[str] = Field(default_factory=list)   # 실제로 포함된 단계(sql/eda/analysis/insight)
    fallback_used: bool = False                              # 숫자 검증/구조 검증 실패로 템플릿 폴백했는지

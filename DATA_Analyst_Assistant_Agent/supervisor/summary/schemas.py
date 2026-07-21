"""노드 서머리 결과 계약 — SQL/EDA/분석/인사이트 4종 노드가 서로 다른 의미적 구조로 나온다.

각 에이전트는 서로 다른 역할을 한다(SQL=정합성 확인·파생변수 생성·마트 설계,
EDA=컬럼 프로파일링·차트 생성, 분석=방법론 선택·가설 검정, 인사이트=근거 종합·직답).
그래서 detail 필드는 노드 종류(kind)별로 다른 스키마를 쓴다(discriminated union) — 같은
종류의 노드는 항상 같은 필드 구조로 나오고(EDA는 항상 EDASummaryDetail), 값만 매 실행마다
달라진다.

title/subtitle/key_finding은 트리에서 노드가 접혀있을 때 보일 공통 요약이고, background/
conclusion도 "왜 이 단계가 필요했는지/무슨 결론을 얻었는지"는 노드 종류 상관없이 같은
역할이라 공통으로 둔다. UI는 background → detail(종류별 섹션) → conclusion 순으로
렌더링한다. FindingSection이 있는 자리는 여전히 heading+본문+차트 반복 패턴을 쓴다.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


class FindingSection(BaseModel):
    """방법 선택의 이유와 관찰 결과를 실제 근거에 묶는 반복 단위."""

    heading: str
    rationale: str = ""                              # 왜 이 방법/차트를 사용했는지
    body: str                                          # 해당 근거에서 무엇을 관찰했는지
    source_label: str | None = None                   # "분포 차트/요약 테이블" 같은 짧은 카테고리 태그
    chart_artifact_ids: list[str] = Field(default_factory=list)   # 복수형 — 한 섹션에 차트 여러 장 가능


class EvidenceTable(BaseModel):
    """분석 결과에 포함된 실제 근거표. UI/Markdown에는 최대 5행만 노출한다."""

    title: str
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)


class SQLSummaryDetail(BaseModel):
    """SQL 단계 — 정합성 확인 → 파생변수 생성 → 데이터마트 설계라는 고유 역할을 반영."""

    kind: Literal["sql"] = "sql"
    design_rationale: str = ""                         # grain/조인/집계를 이렇게 정한 이유
    source_tables: list[str] = Field(default_factory=list)     # 사용한 원천 테이블
    integrity_checks: list[str] = Field(default_factory=list)  # 정합성/결측/중복 처리 항목
    derived_columns: list[FindingSection] = Field(default_factory=list)  # 파생변수 각각(정의·계산식)
    mart_grain: str = ""                                 # 최종 마트의 grain(행 단위)
    mart_columns: list[str] = Field(default_factory=list)      # 최종 마트 컬럼 목록
    mart_preview: list[dict[str, Any]] = Field(default_factory=list)  # 실제 CSV 앞 10행(근거 그대로, LLM이 안 씀)
    sql_snippet: str = ""                                # 실행된 SQL(근거 그대로, LLM이 안 씀)
    interpretation_scope: list[str] = Field(default_factory=list)
    handoff: str = ""                                  # 다음 EDA 단계가 이어받을 분석 가능 범위


class EDASummaryDetail(BaseModel):
    """EDA 단계 — 컬럼 확인 → 통계 탐색 → 차트 생성 → 가설 형성이라는 고유 역할을 반영."""

    kind: Literal["eda"] = "eda"
    data_profile: str = ""                               # 컬럼 타입/카디널리티/결측 요약
    quality_issues: list[str] = Field(default_factory=list)    # 발견된 품질 caution
    statistical_findings: list[FindingSection] = Field(default_factory=list)  # 분포/상관/그룹비교
    hypotheses: list[str] = Field(default_factory=list)        # 제안된 가설 텍스트 요약
    primary_hypothesis: dict[str, Any] = Field(default_factory=dict)  # 근거 그대로 발췌(LLM이 안 씀)
    charts_generated: list[FindingSection] = Field(default_factory=list)  # 구버전 호환용, 신규 생성에서는 비움
    interpretation_scope: list[str] = Field(default_factory=list)
    handoff: str = ""                                  # 분석 단계에서 검증할 가설/조건


class AnalysisSummaryDetail(BaseModel):
    """분석 단계 — 방법론 선택 → 가설 검정이라는 고유 역할을 반영."""

    kind: Literal["analysis"] = "analysis"
    method_decision: dict[str, Any] = Field(default_factory=dict)   # 근거 그대로 발췌(LLM이 안 씀)
    hypothesis_tests: list[FindingSection] = Field(default_factory=list)   # H0/H1/판정/근거
    key_statistics: list[FindingSection] = Field(default_factory=list)    # 핵심 수치 근거
    evidence_tables: list[EvidenceTable] = Field(default_factory=list)    # 원본 analysis_result의 실제 표
    supporting_charts: list[FindingSection] = Field(default_factory=list) # 분석이 실제로 읽은 차트
    interpretation: str = ""                            # 검정들을 함께 읽었을 때의 의미
    limitations: list[str] = Field(default_factory=list)
    handoff: str = ""                                  # 인사이트 단계에서 채택할 결론과 해석 경계


class InsightSummaryDetail(BaseModel):
    """인사이트 단계 — 상류 근거 종합 → 사용자 질문 직답이라는 고유 역할을 반영."""

    kind: Literal["insight"] = "insight"
    evidence_synthesis: str = ""                        # 어떤 상류 근거가 결론을 지지하는지
    answer: str = ""
    key_insights: list[str] = Field(default_factory=list)
    action_plan: list[str] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)   # 사람이 읽는 출처 라벨(근거 그대로)
    limitations: list[str] = Field(default_factory=list)
    supporting_charts: list[FindingSection] = Field(default_factory=list)  # 근거 그대로(LLM이 안 씀)


NodeSummaryDetail = Annotated[
    Union[SQLSummaryDetail, EDASummaryDetail, AnalysisSummaryDetail, InsightSummaryDetail],
    Field(discriminator="kind"),
]


class NodeSummaryResult(BaseModel):
    """노드 하나(아티팩트 1개+)를 설명하는 서머리. fallback 경로에서도 전부 채워진다."""

    title: str
    subtitle: str                                      # 제목 아래 한 줄 태그라인
    background: str                                     # 왜 이 단계가 필요했는지(목적/배경), 최대 2문단
    code_used: str = ""                                # UI 규약: ""면 코드 섹션 자체를 숨긴다
    detail: NodeSummaryDetail
    conclusion: str                                     # "이 단계에서 얻은 결론"
    key_finding: str                                    # 트리에서 노드가 접혀있을 때 보일 짧은 한 줄
    source_kind: str                                    # 근거 아티팩트 종류(sql_plan/eda_summary/...)
    fallback_used: bool = False                          # 숫자 검증 실패로 템플릿 폴백했는지

"""노드 서머리 결과 계약 — SQL/EDA/분석/인사이트 4종 노드가 전부 같은 모양으로 나온다.

UI는 "단계형 상세 패널" — title/subtitle 아래 background(왜 이 단계인지) →
checked_items(뭘 확인했는지) → findings(소제목+본문+차트 반복, 핵심 파트) → conclusion,
이 순서로 렌더링된다. findings 섹션마다 chart_artifact_ids가 있으면 본문 옆/아래에
이미지를 끼워넣는다(차트가 맨 아래 몰리지 않고 관련 섹션 안에 들어가게).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FindingSection(BaseModel):
    """findings 리스트의 항목 하나 — 소제목 + 본문 + (있으면) 관련 차트들."""

    heading: str
    body: str
    source_label: str | None = None                   # "분포 차트"/"요약 테이블" 같은 짧은 카테고리 태그
    chart_artifact_ids: list[str] = Field(default_factory=list)   # 복수형 — 한 섹션에 차트 여러 장 가능


class NodeSummaryResult(BaseModel):
    """노드 하나(아티팩트 1개+)를 설명하는 서머리. fallback 경로에서도 전부 채워진다."""

    title: str
    subtitle: str                                      # 제목 아래 한 줄 태그라인
    background: str                                     # 왜 이 단계가 필요했는지(목적/배경), 최대 2문단
    checked_items: list[str] = Field(default_factory=list)   # "실제로 확인한 항목" 불릿, 4~6개
    code_used: str = ""                                # 근거에서 그대로 가져온 코드/SQL(LLM이 쓰지 않음).
                                                        # UI 규약: ""면 코드 섹션 자체를 숨긴다(insight
                                                        # 노드는 직접 코드를 안 짜서 빈 값이 흔함).
    findings: list[FindingSection] = Field(default_factory=list)
    conclusion: str                                     # "이 단계에서 얻은 결론"
    key_finding: str                                    # 트리에서 노드가 접혀있을 때 보일 짧은 한 줄
    source_kind: str                                    # 근거 아티팩트 종류(sql_result/eda_summary/...)
    fallback_used: bool = False                          # 숫자 검증 실패로 템플릿 폴백했는지

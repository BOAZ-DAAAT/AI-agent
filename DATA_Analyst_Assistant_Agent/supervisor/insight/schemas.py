"""Insight Agent 계약 — 도구 호출(LLM 신호)과 최종 산출물 스키마.

LLM은 매 라운드 ToolCall(JSON)로 '제안'만 하고, 실행·검증은 코드가 한다
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# LLM이 고를 수 있는 행동 전부. 검증(verify)은 여기 없다 — finish에 붙는 강제 게이트라
# LLM이 생략을 '선택'할 수 없다.
TOOLS = ("look", "compute", "chart", "finish")


class ToolCall(BaseModel):
    """LLM이 매 라운드 발행하는 구조적 신호(자유 문자열 실행 아님)."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""                                  # 왜 이 행동인지 — steps 트레이스에 남음


class ChartEntry(BaseModel):
    """렌더된 차트 1장. artifact_id는 등록 성공 시 채움(EDA와 같은 가드 패턴 — 실패해도 안 막음)."""

    filename: str
    title: str = ""
    kind: str = ""                                    # line | bar | table
    supports: str = "answer"                          # 이 차트가 뒷받침하는 대상
    artifact_id: str | None = None
    local_path: str = ""


class InsightResult(BaseModel):
    """루프(loop.py) 결과 — agent.py가 아티팩트로 포장한다."""

    answer: str = ""                                  # 사용자 질문 직답 문장
    key_insights: list[str] = Field(default_factory=list)
    action_plan: list[str] = Field(default_factory=list)   # 근거 없으면 빈 채로가 정직
    limitations: list[str] = Field(default_factory=list)
    charts: list[ChartEntry] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)  # 행동 트레이스(provenance)
    fallback_used: bool = False                       # 검증 실패/라운드 소진 → 보수 답변
    rounds: int = 0

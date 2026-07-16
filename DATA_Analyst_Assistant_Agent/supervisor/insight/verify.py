"""finish 강제 게이트 — 답변 속 숫자가 증거에 실존하는지 결정론 검증.

LLM이 고를 수 있는 도구가 아니다: finish 는 반드시 여길 통과해야 종료된다
(codegen 게이트를 LLM이 선택하지 않는 것과 같은 원칙 — 검증 생략은 선택지가 아님).
실패하면 없는 숫자 목록이 루프 피드백으로 돌아간다.

매칭 규칙(관대하되 지어낸 값은 잡는다):
  ① 리터럴 — 토큰이 증거 corpus 문자열에 그대로 존재 (연도·ID·정확값)
  ② 반올림 — 증거값을 주장 자릿수로 반올림하면 일치 (0.8451 → "0.845")
  ③ 퍼센트 — 증거의 비율(0.418)을 %로 쓴 경우 (→ "41.8%")
소형 정수(≤12, % 아님)는 서수/개수 표현("상위 10", "3분기")이라 검증 제외.
"""

from __future__ import annotations

import json
import re
from typing import Any

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_SMALL_INT_SKIP = 12                                  # "상위 10개"류 서수 허용 상한


def collect_numbers(obj: Any, out: set | None = None) -> set:
    """dict/list/스칼라를 재귀로 훑어 수치만 모은다(검증 대조군)."""
    if out is None:
        out = set()
    if isinstance(obj, bool):                          # True/False 는 숫자 아님
        return out
    if isinstance(obj, (int, float)):
        try:
            out.add(float(obj))
        except (OverflowError, ValueError):
            pass
    elif isinstance(obj, dict):
        for v in obj.values():
            collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_numbers(v, out)
    return out


def build_evidence_corpus(pack, computes: list[Any]) -> tuple[set, str]:
    """증거 숫자 집합 + 리터럴 검색용 문자열을 만든다.

    df 원본 전체는 넣지 않는다(요약·look·compute 로 본 값만 인용 가능) —
    '봤다고 말할 수 있는 것만 쓴다'가 정책이다.
    """
    sources = [pack.table_summary, pack.eda, pack.analysis, computes]
    numbers: set = set()
    for src in sources:
        collect_numbers(src, numbers)
    corpus = json.dumps(sources, ensure_ascii=False, default=str)
    return numbers, corpus


def extract_claims(text: str) -> list[tuple[str, float, int]]:
    """문장에서 검증 대상 수치 토큰을 뽑는다. 반환: (토큰, 값, 주장 소수 자릿수)."""
    claims: list[tuple[str, float, int]] = []
    for m in _NUM_RE.finditer(text or ""):
        token = m.group()
        cleaned = token.replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        decimals = len(cleaned.split(".")[1]) if "." in cleaned else 0
        is_percent = text[m.end():m.end() + 1] in {"%", "％"}
        # 서수/개수용 소형 정수는 스킵 — 단 "10%" 처럼 %가 붙으면 주장이므로 검증
        if decimals == 0 and abs(value) <= _SMALL_INT_SKIP and not is_percent:
            continue
        claims.append((token, value, decimals))
    return claims


def _matches(value: float, decimals: int, evidence_numbers: set) -> bool:
    for e in evidence_numbers:
        try:
            if round(e, decimals) == value:            # 반올림 일치
                return True
            if round(e * 100, decimals) == value:      # 비율 → % 표기
                return True
            if round(e / 100, decimals) == value:      # % 값 → 비율 표기
                return True
        except (OverflowError, ValueError):
            continue
    return False


def verify_texts(texts: list[str], evidence_numbers: set, corpus: str) -> tuple[bool, list[str]]:
    """답변 문장들의 수치가 전부 증거에 근거하는지 판정. 반환: (통과, 근거 없는 토큰들)."""
    missing: list[str] = []
    for text in texts:
        for token, value, decimals in extract_claims(text or ""):
            literal = re.search(rf"(?<![\d.]){re.escape(token.replace(',', ''))}(?!\d)", corpus)
            if literal:
                continue
            if _matches(value, decimals, evidence_numbers):
                continue
            missing.append(token)
    return (not missing), missing

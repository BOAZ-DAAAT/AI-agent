"""숫자 환각 차단 — LLM 답변 속 숫자가 증거에 실존하는지 결정론 검증(공용 유틸).

insight·summary 등 "근거를 읽고 글을 쓰는" 생성기들이 공유한다. LLM이 지어낸 숫자를
코드로 대조해 걸러내는 순수 함수 모음(특정 에이전트에 의존하지 않음).
실패하면 근거에 없는 숫자 토큰 목록을 돌려줘 호출부가 재작성/폴백을 결정하게 한다.

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

_NUM_RE = re.compile(r"(?<!\d)-?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?")
_SMALL_INT_SKIP = 12                                  # "상위 10개"류 서수 허용 상한
_SCIENTIFIC_ABS_THRESHOLD = 1e-4                      # 이 이하 절대값은 반올림 비교가 늘 0으로 붕괴돼 상대오차로 비교


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
    """문장에서 검증 대상 수치 토큰을 뽑는다. 반환: (토큰, 값, 주장 소수 자릿수).

    "7.76e-315" 같은 과학적 표기는 정규식이 지수부까지 통째로 잡으므로, 자릿수(decimals)는
    지수부를 뺀 가수(mantissa) 부분만 보고 계산한다 — 아니면 "e-315"의 "-315"를 소수
    자릿수로 오인하거나, float() 이전에 토큰이 쪼개져 지수부가 별도 숫자로 취급된다.
    """
    claims: list[tuple[str, float, int]] = []
    for m in _NUM_RE.finditer(text or ""):
        token = m.group()
        cleaned = token.replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        mantissa = re.split(r"[eE]", cleaned)[0]
        decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
        is_percent = text[m.end():m.end() + 1] in {"%", "％"}
        is_scientific = "e" in cleaned.lower()
        # 서수/개수용 소형 정수는 스킵 — 단 "10%"처럼 %가 붙거나 과학적 표기면 주장이므로 검증
        if decimals == 0 and abs(value) <= _SMALL_INT_SKIP and not is_percent and not is_scientific:
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
        # p=7.76e-315처럼 절대값이 극히 작은 값은 자릿수 반올림이 항상 0으로 붕괴되므로,
        # 가수(mantissa) 정밀도에 맞춘 상대오차로 대신 비교한다.
        if (
            value != 0
            and e != 0
            and (abs(value) < _SCIENTIFIC_ABS_THRESHOLD or abs(e) < _SCIENTIFIC_ABS_THRESHOLD)
        ):
            try:
                if abs((value - e) / e) <= 0.5 * (10 ** -decimals):
                    return True
            except (OverflowError, ValueError, ZeroDivisionError):
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

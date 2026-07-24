"""구조화 JSON 생성이 실패했을 때 형식 요구를 낮춰 LLM에 한 번 더 맡기는 공용 헬퍼.

report/generator.py, summary/generator.py 가 함께 쓴다(2026-07-23) — JSON 스키마
검증에 걸려 곧장 "자동 생성에 실패했습니다" 정적 템플릿으로 떨어지느니, 라벨 몇 개짜리
순수 텍스트 형식(구조 요구가 낮아 실패 확률이 훨씬 낮다)으로 같은 근거를 한 번 더
서술시켜 실제로 생성된 문장을 확보한다.

호출부는 서사 라벨(제목/한줄요약/결론 등)뿐 아니라 데이터 섹션 라벨(예: fact 키 하나당
하나씩)도 같은 line_specs 에 같이 넣어서 요청한다 — 그래야 "사람이 쓴 서사 + 기계가
덤프한 raw 값"처럼 톤이 갈라지지 않고, 문서 전체가 한 번의 호출·같은 목소리로 나온다.
라벨을 일부 못 지켜도(LLM이 빼먹어도) 호출부가 그 라벨만 원본 값으로 메우면 되므로,
여기서는 파싱된 것만 관대하게 돌려준다 — 전부 못 지켜도 통짜 텍스트는 보존한다.
"""

from __future__ import annotations

import re
from typing import Any

_STYLE_GUIDE = (
    '[문체] 문장 끝을 전부 "~하였습니다/~합니다/~습니다"로 반복하지 마라. 어미를 다양하게 '
    '섞어라(예: "~로 나타났다", "~것으로 보인다", "~점이 확인됐다", "~와 맞물려 있다"). '
    '"~을 진행하여 ~을 확인하였습니다" 같은 상투적인 보고서 문구는 피하고, 실제 분석가가 '
    "동료에게 결과를 설명하듯 자연스럽게 써라. 아래 각 라벨도 전부 같은 목소리로 이어지게 "
    "써라 — 사실을 나열하는 게 아니라 하나의 글처럼."
)


def generate_plain_narrative(
    llm: Any,
    *,
    label: str,
    line_specs: list[tuple[str, str]],
    evidence_text: str,
    charts_text: str,
) -> dict[str, str] | None:
    """line_specs: [(라벨, 힌트), ...] — 프롬프트에 이 순서 그대로 라인이 된다.

    반환값은 {라벨: 텍스트} 딕셔너리(매칭 안 된 라벨은 키 자체가 없음). LLM 호출이
    실패하거나(예외) 응답이 완전히 비어 있으면 None을 돌려준다.
    """
    lines = "\n".join(f"{lbl}: ({hint})" for lbl, hint in line_specs)
    prompt = f"""아래는 '{label}' 단계가 실제로 만든 근거 데이터다. 이 내용을 읽고 있었던 일을
자연스러운 한국어로 정리하라. JSON이나 코드블록, 특수 기호 없이, 아래 라벨만 그대로 남기고
각 줄을 채워라. 근거에 있는 숫자와 사실만 써라 — 없는 숫자를 지어내지 마라.

{_STYLE_GUIDE}

[근거]
{evidence_text}

[관련 차트]
{charts_text}

{lines}"""
    try:
        raw = llm.invoke(prompt).content
    except Exception:  # noqa: BLE001 — 이것도 실패하면 호출부가 최종 템플릿으로 간다
        return None
    if not (raw or "").strip():
        return None
    labels = [lbl for lbl, _ in line_specs]
    return _parse_labeled_text(raw, labels)


def _parse_labeled_text(raw: str, labels: list[str]) -> dict[str, str]:
    text = (raw or "").strip()
    result: dict[str, str] = {}
    # 라벨을 긴 것부터 매칭해야 겹치는 라벨(예: "결론"이 "결론 및 제언"의 부분 문자열인
    # 경우)이 잘못 잘리지 않는다.
    ordered = sorted(set(labels), key=len, reverse=True)
    pattern = "|".join(re.escape(lbl) for lbl in ordered)
    matches = list(re.finditer(rf"\[?({pattern})\]?\s*[:：]\s*", text))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        value = text[start:end].strip()
        if value:
            result[m.group(1)] = value
    if not result and text:
        result["__all__"] = text          # 라벨을 하나도 못 지켰어도 통짜 텍스트는 보존
    return result

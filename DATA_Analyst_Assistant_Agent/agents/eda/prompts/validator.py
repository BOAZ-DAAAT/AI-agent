"""validator(자기검증) 프롬프트.

EDA 산출물(분석·인사이트·가설·요약)이 '다음 단계로 넘겨도 되는지'를 판단하는 관문.
완벽을 요구하지 않는다 — '명백한 결함'이 있을 때만 retry, 아니면 pass(기본).
문제가 있으면 어디를 고쳐야 하는지(planner/insight/hypothesis)까지 판정한다.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def validator_prompt(
    user_question: str,
    question_type: str,
    completed_analyses: List[str],
    statistical_metadata: Dict[str, Any],
    insight_result: str,
    hypotheses: str,
    final_summary: str,
) -> str:
    return f"""
너는 EDA 결과의 '관문 검수자'다. 목표는 완벽을 요구하는 게 아니라,
이 결과를 다음 단계로 넘겨도 되는지(질문에 답했고, 명백히 틀린 내용이 없는지)만 판단하는 것이다.
기본 판정은 'pass'다. 아래 '명백한 결함'이 있을 때만 retry 하라.

[사용자 질문] {user_question}
[question_type] {question_type}
[실행된 분석] {completed_analyses}

[검증된 수치 (인사이트의 숫자는 이것과 일치해야 한다)]
{json.dumps(statistical_metadata, ensure_ascii=False, indent=2)}

[인사이트]
{insight_result}

[가설]
{hypotheses}

[최종 요약]
{final_summary}

── retry 해야 하는 '명백한 결함' (이때만 retry) ──
1. 환각/오류: 인사이트의 숫자가 [검증된 수치]와 명백히 다르거나, 없는 값을 지어냄 → retry_target="insight"
2. 핵심 분석 누락: 질문에 답하는 데 꼭 필요한 분석이 빠짐 (예: 비교 질문인데 comparison 안 함) → retry_target="planner"
3. 질문 미응답: 최종 요약이 사용자 질문에 사실상 답하지 않음 → retry_target="insight"
4. 가설 결함: 가설이 비어있거나 검증 불가능/순환논리 → retry_target="hypothesis"

── retry 사유가 '아닌' 것 (이런 건 그냥 pass) ──
- 더 깊이/추가로 분석할 여지가 있음
- 표현을 더 다듬을 수 있음
- 사소한 개선점, "조금 더 보면 좋겠다" 수준
- 한계 언급이 조금 부족한 정도

즉 결과가 '틀리거나 핵심이 빠졌을' 때만 retry. 단지 '완벽하지 않을 뿐'이면 pass 하라.
확신이 없으면 pass 하라.

반드시 아래 JSON만 출력하라. 설명 없이.
{{
  "status": "pass" 또는 "retry",
  "retry_target": "planner" / "insight" / "hypothesis" / "none",
  "reason": "판정 이유 한 줄",
  "feedback": "retry일 때, 해당 노드가 보고 고칠 구체적 지시 (pass면 빈 문자열)"
}}
"""

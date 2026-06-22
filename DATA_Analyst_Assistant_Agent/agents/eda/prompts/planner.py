"""플래너(적응형 컨트롤러) 결정 프롬프트.

매 라운드: 질문 + 데이터 모양 + 지금까지 분석 결과 + 실행 가능한 분석 카드를 보고
다음에 돌릴 분석 1개(또는 "done")를 고른다. 첫 라운드엔 priority_metrics도 정한다.
"""

from __future__ import annotations

from typing import Any, Dict, List


def planner_prompt(
    user_question: str,
    question_type: str,
    data_shape: Dict[str, Any],
    feasible_cards: List[Dict[str, str]],
    completed_findings: Dict[str, str],
    round_idx: int,
    max_rounds: int,
    need_priority: bool,
) -> str:
    cards_text = "\n".join(f"- {c['name']}: {c['desc']}" for c in feasible_cards) or "(없음)"
    if completed_findings:
        done_text = "\n".join(
            f"[{name}] {(summary or '')[:400]}" for name, summary in completed_findings.items()
        )
    else:
        done_text = "(아직 없음 — 첫 라운드)"

    priority_block = ""
    if need_priority:
        priority_block = """
또한 이번이 첫 라운드이므로 priority_metrics(사용자 질문과 가장 관련 높은 measure 컬럼 1~3개)도 함께 정하라.
"""

    priority_json = ',\n  "priority_metrics": [{"metric": "컬럼명", "reason": "왜 핵심인지"}]' if need_priority else ""

    return f"""
너는 EDA 탐색을 지휘하는 분석 전략가다.
사용자 질문에 답하고 검증 가능한 가설을 세우기 위해, 지금 어떤 분석을 다음으로 돌릴지 1개만 고른다.

[사용자 질문] {user_question}
[question_type] {question_type}
[데이터 모양]
- 전체 행 수: {data_shape.get('row_count')}
- 수치형 컬럼: {data_shape.get('numeric_cols')}
- 범주형 컬럼: {data_shape.get('cat_cols')}
- 시간 컬럼: {data_shape.get('time_cols')}

[지금까지 분석한 결과 (이걸 보고 다음을 정하라)]
{done_text}

[지금 실행 가능한 분석 (이 목록에서만 골라라)]
{cards_text}

[현재 {round_idx + 1}번째 라운드 / 최대 {max_rounds}라운드]
{priority_block}
판단 기준:
- 사용자 질문에 답하고 가설을 세우기에 "충분히" 봤다면 "done"을 골라라. 모든 분석을 다 할 필요 없다.
- 지금까지 결과에서 흥미로운 단서(강한 상관, 큰 그룹 차이 등)가 보이면, 그걸 더 파고들 분석을 골라라.
- 실행 가능 목록에 없는 분석은 고르지 마라.

반드시 아래 JSON만 출력하라. 설명 없이.
{{
  "next": "위 목록의 분석명 중 하나, 또는 done",
  "reason": "왜 이걸 골랐는지(또는 왜 done인지) 한 문장"{priority_json}
}}
"""

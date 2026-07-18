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
    codegen_selectable: bool = False,
    emit_batch: bool = False,
) -> str:
    cards_text = "\n".join(f"- {c['name']}: {c['desc']}" for c in feasible_cards) or "(없음)"

    # codegen 카드 — 6개 분석 도구로 구조적으로 계산 불가한 질문에만. 도구 우선(over-fire 방지 울타리).
    codegen_block = ""
    codegen_rule = ""
    codegen_next_hint = ""
    if codegen_selectable:
        codegen_block = (
            "\n\n[특수 카드]\n"
            "- codegen: 위 분석 도구로는 계산할 수 없는 파생 지표를 직접 계산한다 "
            "(예: 조건부 비율 'X 이상 비율', 특정 서브셋 집계 '상위 10% 중 평균', "
            "사용자 정의 파생 컬럼, 그룹별 커스텀 비율/순위)."
        )
        codegen_rule = (
            '\n- ⚠️ codegen은 최후수단이다. 분포·비교·상관·시계열 도구로 답할 수 있는 질문이면 '
            "절대 codegen을 고르지 마라(도구 우선). 위 도구들로 구조적으로 계산이 불가능한 "
            "파생 계산이 질문의 핵심일 때만 codegen을 골라라. "
            "이미 실행한 분석으로 질문에 충분히 답했다면 codegen이 아니라 done을 골라라. "
            "정식 통계검정(t검정, ANOVA, 카이제곱, 회귀모형 적합 등)의 실행은 다음 분석 에이전트의 역할이다. "
            "검정 실행이나 p-value 계산을 위해 codegen을 고르지 말고, EDA 근거가 충분하면 done을 골라라."
        )
        codegen_next_hint = ', codegen'
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

    if not emit_batch:
        goal_line = "사용자 질문에 답하고 검증 가능한 가설을 세우기 위해, 지금 어떤 분석을 다음으로 돌릴지 1개만 고른다."
        batch_block = ""
        output_schema = f"""{{
  "next": "위 목록의 분석명 중 하나, done{codegen_next_hint}",
  "reason": "왜 이걸 골랐는지(또는 왜 done인지) 한 문장"{priority_json}
}}"""
    else:
        # 첫 라운드 전용 — 매 라운드 1개씩 고르며 왕복하는 대신, 한 번에 세트를 정해
        # 이후 라운드는 이 세트를 LLM 호출 없이 순서대로 소비한다(#194, 라운드 왕복 비용 절감).
        goal_line = ("사용자 질문에 답하고 검증 가능한 가설을 세우기 위해 필요한 분석들을 "
                    "이번 한 번에 세트로 정한다(1개씩 왕복하며 고르지 않는다).")
        batch_block = f"""
[배치 계획 — 이번이 첫 라운드다]
- 필요한 분석을 실행 가능 목록에서 골라 "next_batch"에 중요한 순서대로 담아라
  (보통 2~4개, 단순한 질문이면 1개만 담아도 된다. 정말 다 필요하면 더 담아도 된다).
- codegen이 필요하다고 판단되면 next_batch는 빈 배열로 두고 "next"에 "codegen"을 넣어라
  (codegen은 실행 방식이 달라 다른 분석과 한 세트로 못 묶는다).
- 어떤 분석도 필요 없다고 판단되면 next_batch는 빈 배열로 두고 "next"를 "done"으로.
"""
        output_schema = f"""{{
  "next_batch": ["실행할 분석명들을 순서대로", "..."],
  "next": "next_batch를 비웠을 때만 사용 — codegen 또는 done",
  "reason": "왜 이 세트를(또는 codegen/done을) 골랐는지 한 문장"{priority_json}
}}"""

    return f"""
너는 EDA 탐색을 지휘하는 분석 전략가다.
{goal_line}

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
{cards_text}{codegen_block}

[현재 {round_idx + 1}번째 라운드 / 최대 {max_rounds}라운드]
{priority_block}{batch_block}
판단 기준:
- 사용자 질문에 답하고 가설을 세우기에 "충분히" 볼 수 있는 만큼만 골라라. 모든 분석을 다 담을 필요 없다.
- 지금까지 결과에서 흥미로운 단서(강한 상관, 큰 그룹 차이 등)가 보이면, 그걸 더 파고들 분석을 포함하라.
- 실행 가능 목록에 없는 분석은 고르지 마라.{codegen_rule}

반드시 아래 JSON만 출력하라. 설명 없이.
{output_schema}
"""

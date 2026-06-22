"""분석 노드(inspect/quality/distribution/comparison/relationship/time) 프롬프트
+ mini-ReAct 재시도용 수정 프롬프트."""

from __future__ import annotations

from typing import Any, Dict


def inspect_prompt(user_question: str, grain: str) -> str:
    return f"""
너는 데이터 구조 분석 전문가다.

[사용자 쿼리] {user_question}
[grain] {grain}

profile_data를 호출하여:
1. 컬럼 타입과 카디널리티 파악
2. 수치형 / 범주형 / 시간형 컬럼 분류
3. grain 확인
4. 분석 가능한 지표 목록 정리
결과를 한국어로 요약하라.
"""


def quality_prompt(user_question: str, inspect_result: str) -> str:
    return f"""
너는 데이터 품질 전문가다.

[사용자 쿼리] {user_question}
[구조 파악 결과]
{inspect_result}

아래 툴을 모두 호출하여 데이터 품질을 점검하라:
1. check_missing — 결측치
2. check_outliers — 이상치
3. check_duplicates — 중복
4. check_sample_reliability — 표본 수 신뢰도

결과를 종합하여 한국어로 요약하라. 특히 분석 시 주의해야 할 품질 이슈를 명시하라.
"""


def distribution_prompt(user_question: str, inspect_result: str, plan: Dict[str, Any]) -> str:
    return f"""
너는 단변량 분포 분석 전문가다.

[사용자 쿼리] {user_question}
[구조 파악 결과]
{inspect_result}
[이번 분석 집중 전략] {plan.get('distribution_focus', '전체 수치형 컬럼 분포 확인')}
[우선 분석 지표] {plan.get('priority_metrics', [])}

데이터 특성에 맞게 아래 툴 중 필요한 것을 선택하여 호출하라:
- draw_distributions: 수치형 컬럼이 있을 때 (히스토그램)
- draw_boxplots: 이상치 시각화가 필요할 때
- draw_category_distribution: 범주형 컬럼이 있을 때

우선 분석 지표를 중심으로 분포 형태, 치우침, 분산 정도를 한국어로 요약하라.
"""


def comparison_prompt(user_question: str, inspect_result: str, plan: Dict[str, Any]) -> str:
    return f"""
너는 그룹 비교 분석 전문가다.

[사용자 쿼리] {user_question}
[구조 파악 결과]
{inspect_result}
[이번 분석 집중 전략] {plan.get('comparison_focus', '카테고리별 지표 비교')}
[우선 분석 지표] {plan.get('priority_metrics', [])}

아래 툴을 모두 호출하여 그룹 간 비교를 수행하라:
1. draw_top_n_barplot — 지표별 상위/하위 카테고리
2. draw_heatmap_matrix — 전체 카테고리 × 지표 한눈에 비교

우선 분석 지표를 중심으로 어떤 그룹이 강하고 약한지 한국어로 요약하라.
"""


def relationship_prompt(user_question: str, inspect_result: str, plan: Dict[str, Any]) -> str:
    return f"""
너는 변수 관계 분석 전문가다.

[사용자 쿼리] {user_question}
[구조 파악 결과]
{inspect_result}
[이번 분석 집중 전략] {plan.get('relationship_focus', '수치형 지표 간 상관관계 탐색')}
[우선 분석 지표] {plan.get('priority_metrics', [])}

아래 툴을 모두 호출하여 변수 간 관계를 탐색하라:
1. draw_correlation — 수치형 변수 간 상관관계
2. draw_scatter_pairs — 변수 쌍별 관계 시각화

우선 분석 지표와 관련된 상관관계, trade-off, 주목할 패턴을 한국어로 요약하라.
"""


def time_prompt(user_question: str, inspect_result: str) -> str:
    return f"""
너는 시계열 분석 전문가다.

[사용자 쿼리] {user_question}
[구조 파악 결과]
{inspect_result}

아래 툴을 호출하여 시간 패턴을 분석하라:
1. draw_timeseries — 시계열 추세
2. draw_seasonality — 월/요일 시즌성

추세, 계절성, 특이 시점을 한국어로 요약하라.
"""


def react_fix_prompt(node_name: str, last_error: str, current_prompt: str) -> str:
    return f"""
아래 분석 노드({node_name})에서 아래 에러가 발생했다.
에러: {last_error}

원래 분석 지시:
{current_prompt}

에러 원인을 파악하고, 에러를 피할 수 있도록 분석 방식을 조정한 새로운 지시문을 작성하라.
- 존재하지 않는 컬럼 참조나 타입 오류라면 해당 분석을 생략하도록 지시하라.
- 수정된 지시문만 출력하라. 설명 없이.
"""

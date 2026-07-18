"""분석 노드(inspect/quality/distribution/comparison/relationship/time) 프롬프트.

각 노드는 스킬을 코드로 직접 실행한 뒤 그 결과(result_json)를 이미 계산된 값으로
프롬프트에 넣는다 — LLM은 요약만 한다(#194, 예전 mini-ReAct 툴판단 라운드 제거)."""

from __future__ import annotations

from typing import Any, Dict


def inspect_prompt(user_question: str, grain: str, result_json: str) -> str:
    return f"""
너는 데이터 구조 분석 전문가다.

[사용자 쿼리] {user_question}
[grain] {grain}

[데이터 기본 구조 — shape/타입/카디널리티/시간컬럼/기초통계, 이미 계산됨]
{result_json}

위 결과를 바탕으로:
1. 컬럼 타입과 카디널리티 파악
2. 수치형 / 범주형 / 시간형 컬럼 분류
3. grain 확인
4. 분석 가능한 지표 목록 정리
결과를 한국어로 요약하라.
"""


def quality_prompt(user_question: str, inspect_result: str, result_json: str) -> str:
    return f"""
너는 데이터 품질 전문가다.

[사용자 쿼리] {user_question}
[구조 파악 결과]
{inspect_result}

[품질 점검 결과 — 결측치/이상치/중복/표본신뢰도, 이미 계산됨]
{result_json}

위 결과를 종합하여 한국어로 요약하라. 특히 분석 시 주의해야 할 품질 이슈를 명시하라.
"""


def distribution_prompt(user_question: str, inspect_result: str, plan: Dict[str, Any], result_json: str) -> str:
    return f"""
너는 단변량 분포 분석 전문가다.

[사용자 쿼리] {user_question}
[구조 파악 결과]
{inspect_result}
[이번 분석 집중 전략] {plan.get('distribution_focus', '전체 수치형 컬럼 분포 확인')}
[우선 분석 지표] {plan.get('priority_metrics', [])}

[분포 분석 결과 — 히스토그램/박스플롯/범주분포 통계, 이미 계산됨]
{result_json}

우선 분석 지표를 중심으로 분포 형태, 치우침, 분산 정도를 한국어로 요약하라.
"""


def comparison_prompt(user_question: str, inspect_result: str, plan: Dict[str, Any], result_json: str) -> str:
    return f"""
너는 그룹 비교 분석 전문가다.

[사용자 쿼리] {user_question}
[구조 파악 결과]
{inspect_result}
[이번 분석 집중 전략] {plan.get('comparison_focus', '카테고리별 지표 비교')}
[우선 분석 지표] {plan.get('priority_metrics', [])}

[그룹 비교 결과 — 상위/하위 카테고리·히트맵 통계, 이미 계산됨]
{result_json}

우선 분석 지표를 중심으로 어떤 그룹이 강하고 약한지 한국어로 요약하라.
"""


def relationship_prompt(user_question: str, inspect_result: str, plan: Dict[str, Any], result_json: str) -> str:
    return f"""
너는 변수 관계 분석 전문가다.

[사용자 쿼리] {user_question}
[구조 파악 결과]
{inspect_result}
[이번 분석 집중 전략] {plan.get('relationship_focus', '수치형 지표 간 상관관계 탐색')}
[우선 분석 지표] {plan.get('priority_metrics', [])}

[관계 탐색 결과 — 상관계수·scatter 통계, 이미 계산됨]
{result_json}

우선 분석 지표와 관련된 상관관계, trade-off, 주목할 패턴을 한국어로 요약하라.
"""


def time_prompt(user_question: str, inspect_result: str, result_json: str) -> str:
    return f"""
너는 시계열 분석 전문가다.

[사용자 쿼리] {user_question}
[구조 파악 결과]
{inspect_result}

[시계열 분석 결과 — 추세/시즌성 통계, 이미 계산됨]
{result_json}

추세, 계절성, 특이 시점을 한국어로 요약하라.
"""

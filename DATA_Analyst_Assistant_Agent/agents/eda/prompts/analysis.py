"""분석 노드(inspect/quality/distribution/comparison/relationship/time) 프롬프트.

각 노드는 스킬을 코드로 직접 실행한 뒤 그 결과(result_json)를 이미 계산된 값으로
프롬프트에 넣는다 — LLM은 요약만 한다(#194, 예전 mini-ReAct 툴판단 라운드 제거)."""

from __future__ import annotations

from typing import Any, Dict

# 노드가 이 마커 뒤 JSON 배열을 분리해 insight 입력용 facts로 쓴다(#194 후속).
# prompts는 nodes에 의존하면 안 되므로(순환임포트) 여기가 정의 위치이고, tool_runner.py가 이걸 가져간다.
ANALYSIS_FACTS_MARKER = "===ANALYSIS_FACTS==="

_FACTS_BLOCK = f"""
마지막으로, 위 요약을 모두 끝낸 뒤 아래 구분선과 JSON 배열을 정확히 그대로 추가로 출력하라
(이 배열은 insight 단계가 원문 대신 참고하는 핵심 사실 목록이다):
{ANALYSIS_FACTS_MARKER}
["핵심 사실 1(한 문장, 200자 이내, 위 결과에 실제 존재하는 근거만, 원인·인과관계 추정 금지)",
 "핵심 사실 2 (다른 항목과 중복 금지, 필요하면 핵심 수치 포함)"]
2~4개만 작성하라.
"""

_EXPLORATORY_BOUNDARY = """
역할 경계:
- 이 노드는 EDA 탐색 요약만 작성한다. 최종 답변, 인과 판단, 통계적 유의성 판단, 가설 채택/기각은 쓰지 마라.
- 허용 표현: "관찰된다", "패턴이 보인다", "후속 Analysis 단계에서 검정이 필요하다", "해석에 주의가 필요하다".
- 금지 표현: "유의하다", "검증되었다", "확인되었다", "영향을 준다", "채택", "기각",
  "supported", "rejected", "inconclusive", "p-value", "effect size".
"""


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
{_EXPLORATORY_BOUNDARY}
{_FACTS_BLOCK}"""


def quality_prompt(user_question: str, inspect_result: str, result_json: str) -> str:
    return f"""
너는 데이터 품질 전문가다.

[사용자 쿼리] {user_question}
[구조 파악 결과]
{inspect_result}

[품질 점검 결과 — 결측치/이상치/중복/표본신뢰도, 이미 계산됨]
{result_json}

위 결과를 종합하여 한국어로 요약하라. 특히 분석 시 주의해야 할 품질 이슈를 명시하라.
{_EXPLORATORY_BOUNDARY}
{_FACTS_BLOCK}"""


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
{_EXPLORATORY_BOUNDARY}
{_FACTS_BLOCK}"""


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
{_EXPLORATORY_BOUNDARY}
{_FACTS_BLOCK}"""


def relationship_prompt(user_question: str, inspect_result: str, plan: Dict[str, Any], result_json: str) -> str:
    return f"""
너는 변수 관계 분석 전문가다.

[사용자 쿼리] {user_question}
[구조 파악 결과]
{inspect_result}
[이번 분석 집중 전략] {plan.get('relationship_focus', '수치형 지표 간 상관관계 탐색')}
[우선 분석 지표] {plan.get('priority_metrics', [])}

[관계 탐색 결과 — scatter/관계 탐색 통계, 이미 계산됨]
{result_json}

우선 분석 지표와 관련된 방향성, trade-off, 주목할 패턴을 한국어로 요약하라.
상관계수, Spearman/Pearson 계수, p-value, 검정 결과 숫자는 본문과 facts에 쓰지 마라.
예: "배송일 구간이 길어질수록 리뷰 중앙값이 낮아지는 패턴이 관찰된다. 후속 Analysis에서 단조 관계를 검정해야 한다."
처럼 방향성과 후속 검정 필요성만 남겨라.
{_EXPLORATORY_BOUNDARY}
{_FACTS_BLOCK}"""


def time_prompt(user_question: str, inspect_result: str, result_json: str) -> str:
    return f"""
너는 시계열 분석 전문가다.

[사용자 쿼리] {user_question}
[구조 파악 결과]
{inspect_result}

[시계열 분석 결과 — 추세/시즌성 통계, 이미 계산됨]
{result_json}

추세, 계절성, 특이 시점을 한국어로 요약하라.
추세선과 월별 변동은 탐색 패턴으로만 설명하고, 예측·원인·유의성 판단은 쓰지 마라.
{_EXPLORATORY_BOUNDARY}
{_FACTS_BLOCK}"""

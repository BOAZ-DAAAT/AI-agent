"""EDA 에이전트의 상태/IO 모델.

원본 `eda_agent/eda_agent.py` 의 LangGraph `EDAState` 를 분리한 것.
DataFrame 등 런타임 객체는 state 에 싣지 않고 `_runtime.EdaContext` 가 보유한다
(원본의 모듈 전역 `_df` 를 대체).
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class EDAState(TypedDict, total=False):
    # 입력 인터페이스 (SQL Agent → EDA Agent)
    user_question: str          # 사용자 원본 질문
    target_table: str           # "sql_agent.{mart_name}" (현재는 참고용, 데이터는 CSV 아티팩트로 진입)
    mart_design: Dict[str, Any] # grain, key_columns, measure_columns
    question_type: str          # "comparison" | "distribution" | "relationship" | "time"

    # 앞단(supervisor/SQL)이 넘긴 의미 힌트 (없거나 컬럼명이 아닐 수 있음 → 폴백 필요).
    # ※ 현재 supervisor는 "매출"/"월" 같은 자연어/None으로만 채움(state.plan). 컬럼명 보장 X.
    plan_metric: str            # 분석 대상(target) 후보 힌트
    plan_dimension: str         # 그룹/단위(dimension) 힌트
    analysis_target: str        # 실제 df 컬럼으로 확정된 target (가설 6유형 앵커)

    # planner 결정 (하위호환 — 컨트롤러가 priority_metrics/focus를 여기 보관)
    analysis_plan: Dict[str, Any]

    # 컨트롤러(플래너) 루프 상태
    round: int                       # 현재 라운드 (0부터)
    next_analysis: str               # 플래너가 고른 다음 분석 ("done" 가능)
    controller_log: List[Dict[str, Any]]  # [{round, choice, reason}] 결정 추적

    # validator(자기검증) 상태
    validation_result: Dict[str, Any]     # {status, retry_target, reason, feedback}
    validation_retries: int               # 검증 재시도 횟수
    validation_feedback: str              # 재시도 대상 노드에 전달할 보완 지시

    # 각 노드 결과
    inspect_result: str
    quality_result: str
    distribution_result: str
    comparison_result: str
    relationship_result: str
    time_result: str
    clustering_result: Dict[str, Any]

    # 컬럼 의미 분류 (LLM이 로드 직후 판단)
    time_columns: List[str]    # 시간/날짜 컬럼
    count_column: str          # 표본 수/볼륨 컬럼 (없으면 "")

    # 플래그
    has_time_column: bool

    # 데이터 한계 자가점검 (insight 노드, 코드 기반)
    data_level: Dict[str, Any]            # {level: raw|aggregated|..., grain_hint, reason}
    cautions: List[str]                   # 해석 주의사항(집계한계·표본부족·상관≠인과 등)

    # 최종 출력
    insight_result: str
    hypotheses: str
    final_summary: str
    key_charts: List[str]
    statistical_metadata: Dict[str, Any]  # downstream 에이전트용 raw 수치
    chart_requests: List[Dict[str, Any]]  # EDA가 발행한 차트 주문서(intent/stats/columns/hint) — chart/ 렌더용

    # 에러 로그
    error_log: List[str]

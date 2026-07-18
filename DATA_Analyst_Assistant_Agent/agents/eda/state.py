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

    # 앞단(오케스트레이터/SQL)이 넘긴 의미 힌트 (없거나 컬럼명이 아닐 수 있음 → 폴백 필요).
    # ※ 앞단은 "매출"/"월" 같은 자연어/None으로만 채우는 경우가 있음(state.plan). 컬럼명 보장 X.
    plan_metric: str            # 분석 대상(target) 후보 힌트
    plan_dimension: str         # 그룹/단위(dimension) 힌트
    # SQL 에이전트가 넘긴 원천 테이블/선언 grain. GE 정합성 스코핑 + grain 교차검증용(없으면 폴백).
    plan_source_tables: List[str]  # GE 정합성을 이 테이블들로 스코핑해 읽는다
    plan_business_grain: str       # SQL이 선언한 엔티티 grain (예: "one row per customer_unique_id")
    analysis_target: str        # 실제 df 컬럼으로 확정된 target (가설 6유형 앵커)

    # planner 결정 (하위호환 — 컨트롤러가 priority_metrics/focus를 여기 보관)
    analysis_plan: Dict[str, Any]

    # 컨트롤러(플래너) 루프 상태
    round: int                       # 현재 라운드 (0부터)
    next_analysis: str               # 플래너가 고른 다음 분석 ("done" 가능)
    controller_log: List[Dict[str, Any]]  # [{round, choice, reason}] 결정 추적
    analysis_queue: List[str]        # 1차 배치계획 중 아직 안 돌린 것들 (#194, 있으면 LLM 호출 없이 소비)

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

    # 각 노드가 원문과 별도로 뽑은 짧은 핵심 사실 — insight 입력용(원문 대체 아니라 병행, #194 후속)
    inspect_facts: List[str]
    quality_facts: List[str]
    distribution_facts: List[str]
    comparison_facts: List[str]
    relationship_facts: List[str]
    time_facts: List[str]

    # 컬럼 의미 분류 (LLM이 로드 직후 판단)
    time_columns: List[str]    # 시간/날짜 컬럼
    count_column: str          # 표본 수/볼륨 컬럼 (없으면 "")

    # 플래그
    has_time_column: bool

    # 데이터 한계 자가점검 (insight 노드, 코드 기반)
    data_level: Dict[str, Any]            # {level, grain_hint, reason, is_aggregated, raw_observation_level_available}
    cautions: List[Dict[str, Any]]        # 계약형 주의사항 구조체(code·severity·message_ko·recommended_action·source)
    analysis_constraints: List[Dict[str, Any]]  # rule 기반 hard 계약(allowed/blocked_operations·reason_ko·unless)

    # 최종 출력
    insight_result: str
    summary_facts: List[str]              # insight가 뽑은 짧은 핵심 사실 문장 — 최종 요약 입력용(원문 재요약 대신, #194)
    hypotheses: str
    hypothesis_signals: List[Dict[str, Any]]  # 가설별 사후 재검증 신호(강도·matched_signal·score·drop_reason) — 설명가능성/디버깅용
    primary_hypothesis: Dict[str, Any]    # 1순위 가설 {target, feature, method} — 최종 요약 입력용(#194)
    final_summary: str
    key_charts: List[str]
    key_chart_captions: Dict[str, str]    # {파일명: 선정 이유 캡션} — 아티팩트 메타데이터로 실림(#71 B)
    statistical_metadata: Dict[str, Any]  # downstream 에이전트용 raw 수치
    chart_requests: List[Dict[str, Any]]  # EDA가 발행한 차트 주문서(intent/stats/columns/hint) — chart/ 렌더용

    # codegen 탈출구 결과 (도구로 답 못 낸 도메인 밖 질문에만 채워짐)
    # {status: "success"|"out_of_domain", ...} — status 필드는 미래 "partial" 확장 여지
    codegen: Dict[str, Any]

    # 에러 로그
    error_log: List[str]

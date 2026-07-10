"""질문 분석(plan_question) 프롬프트."""

from __future__ import annotations


def plan_prompt(state) -> str:
    return f"""
너는 MySQL 기반 SQL/데이터마트 planner다.

사용자 질문:
{state['user_question']}

스키마 JSON:
{state['schema_text']}

정합성 점검 JSON:
{state['integrity_text']}

## route_kind 판단 기준

**route_kind는 반드시 "simple" 또는 "comprehensive" 중 하나다.**

- **simple**: SELECT 문으로 결과를 바로 조회하여 반환할 수 있는 단편적이고 단순한 조회 질문
  - 단순 목록 조회, 특정 조건의 단일 레코드 확인 등 분석적 추세나 깊은 탐색이 필요 없는 단순 조회
  - 예: "상위 10명 판매자 조회해줘", "특정 회원 ID의 가입일 조회", "카테고리 목록 출력", "가장 비싼 상품 5개 조회" 등은 simple

- **comprehensive**: 분석적 깊이가 있거나, 시계열 추이 분석, 혹은 다각적인 비즈니스 분석이 필요하여 분석용 데이터마트(CREATE TABLE AS SELECT)를 구축하고 이를 기반으로 분석하는 것이 적절한 경우
  - 사용자가 명시적으로 "데이터마트", "마트 생성", "분석용 테이블 만들어줘" 등을 요청한 경우
  - 또는 일별/월별 매출 추이 등 추세 분석, 카테고리별 성과 분석, 복잡한 다중 조인(Multi-join) 및 다단계 CTE(Common Table Expression) 분석, 복잡한 비즈니스 로직(예: 코호트 분석, LTV 분석, 리텐션 분석, 고객 세그먼트별 다차원 교차 분석 등)이 포함되어 깊은 분석이 필요한 경우

## task_type 규칙
- route_kind=comprehensive → task_type=data_mart_build
- route_kind=simple → task_type=query_answer

## selected_join_tables 선택 (핵심)

스키마에 실제 존재하는 테이블명만 사용. 질문에 이름이 없어도 의도상 필요하면 포함.

- "매출", "수익", "revenue", "sales" → orders, order_items 우선 (customers 불필요)
- "일별/월별 추이" → 날짜 컬럼이 있는 orders 우선
- "고객 수", "고객별 분석"처럼 고객이 분석 주체일 때만 customers 포함
- 불필요한 테이블은 제외

## expected_result_shape 판단

- GROUP BY + 날짜/카테고리 집계 → "grouped_aggregate"
- 단일 집계 숫자 하나 (예: 평균 배송일, 총 주문수) → "single_scalar"
- 데이터마트 생성 → "datamart_creation"
- 단순 행 조회 → "table_preview"

## 공통 규칙
- relevant_tables와 selected_join_tables는 스키마에 실제 존재하는 테이블만
- 질문에 없는 조건 임의 추가 금지
- 애매한 점은 ambiguity_note에 기록
- 반드시 JSON만 출력

## 출력 형식
{{
  "original_question": "...",
  "question_type": "aggregation / comparison / ranking / filter / detail / trend / identification / mart_build 중 하나",
  "task_type": "query_answer 또는 data_mart_build",
  "requested_output": "sql_only / execute_and_answer / create_table 중 하나",
  "route_kind": "simple 또는 comprehensive",
  "target_metric": "핵심 지표명 (예: 일별 매출합계)",
  "dimensions": ["일", "월", ...],
  "filters": ["조건1", ...],
  "time_condition": "예: 2017년 6월 또는 null",
  "selected_join_tables": ["실제 테이블명", ...],
  "relevant_tables": ["실제 테이블명", ...],
  "expected_result_shape": "single_scalar / grouped_aggregate / table_preview / datamart_creation 중 하나",
  "required_aggregations": ["SUM", "COUNT", ...],
  "mart_name": "... 또는 null",
  "grain": "... 또는 null",
  "load_strategy": "... 또는 null",
  "ambiguity_note": "... 또는 null",
  "reasoning": "테이블 선택, route 판단, expected_result_shape 근거를 한 문장으로"
}}
"""

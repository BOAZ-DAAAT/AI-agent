"""마트 설계(design_mart) 프롬프트."""

from __future__ import annotations

import json

from DATA_Analyst_Assistant_Agent.agents.sql._runtime import ALLOWED_MART_SCHEMA


def mart_design_prompt(state) -> str:
    return f"""
너는 분석용 데이터마트 설계자다.

사용자 질문:
{state['user_question']}

질문 분석 결과:
{json.dumps(state['plan'], ensure_ascii=False, indent=2)}

스키마 JSON:
{state['schema_text']}

정합성 점검 JSON:
{state['integrity_text']}

설계 규칙:
- 분석에 재사용 가능한 데이터마트 기준으로 설계
- 확정된 selected_join_tables, required_columns, business_keys를 먼저 확인
- 각 source_tables의 실제 행 grain 키를 source_grains에 정의
- 모든 target_metrics를 후속 계산할 수 있는 가장 세밀한 공통 분석 grain을 grain과 grain_columns에 정의
- 원천 최저 grain을 무조건 보존하지 말고, 더 세밀한 원천은 공통 grain으로 집계하거나 선행 중복 제거
- deduplication_keys는 grain_columns와 순서까지 완전히 동일해야 함
- column_plan은 최종 출력 순서이며 output_column 중복 금지
- calculation_rule은 SQL 조각이 아니라 컬럼의 자연어 의미와 계산 계약으로 작성
- role은 dimension / measure / attribute 중 하나만 사용
- aggregation_method는 none / SUM / COUNT / COUNT_DISTINCT / MIN / MAX / AVG / DEDUPLICATE 중 하나만 사용
- 모든 aggregation_method가 none일 때만 preserve_common_grain 사용
- 하나라도 집계 또는 DEDUPLICATE가 필요하면 aggregate_to_common_grain 사용
- 질문별 최종 비율, 순위, 최종 판정값, 카테고리 요약 지표는 column_plan에 포함하지 않음
- 배송 지연 산정 대상 여부, 배송 지연 여부, 지연 일수처럼 후속 지표에 필요한 원자적 파생값은 포함 가능
- metric_support는 고유 target_metrics를 주어진 순서로 정확히 한 번씩 포함
- metric_support.calculation_grain은 최종 지표 출력 grain이며 전역 지표만 빈 목록 허용
- 중간 grain과 다단계 계산 순서는 downstream_calculation에 자연어로 기록
- metric_support가 참조하는 calculation_grain과 required_mart_columns는 모두 column_plan의 output_column이어야 함
- 최종 계획의 required_columns를 마트 컬럼 선택의 우선 근거로 사용
- 최종 계획의 business_keys를 조인과 key_columns 선택의 우선 근거로 사용
- target_schema는 "{ALLOWED_MART_SCHEMA}" 로 고정
- incremental이 자연스러우면 incremental_column 제안
- 질문에 없는 정의를 과도하게 추가하지 말고 reasoning에 근거 설명
- 반드시 JSON만 출력

출력 형식:
{{
  "mart_name": "...",
  "target_schema": "{ALLOWED_MART_SCHEMA}",
  "grain": "customer_unique_id × order_id × category처럼 자연어로 쓴 공통 분석 grain",
  "grain_columns": ["customer_unique_id", "order_id", "category"],
  "source_tables": ["..."],
  "source_grains": {{"orders": ["order_id"], "order_items": ["order_id", "order_item_id"]}},
  "deduplication_keys": ["customer_unique_id", "order_id", "category"],
  "column_plan": [
    {{
      "output_column": "...",
      "role": "dimension 또는 measure 또는 attribute",
      "source_columns": ["table.column"],
      "calculation_type": "passthrough 또는 derived",
      "calculation_rule": "SQL이 아닌 자연어 의미 계약",
      "aggregation_method": "none 또는 SUM 또는 COUNT 또는 COUNT_DISTINCT 또는 MIN 또는 MAX 또는 AVG 또는 DEDUPLICATE",
      "inclusion_reason": "..."
    }}
  ],
  "metric_support": [
    {{
      "metric_name": "target_metrics의 항목",
      "calculation_grain": ["최종 지표 출력 grain 컬럼"],
      "required_mart_columns": ["column_plan에 선언된 컬럼"],
      "downstream_calculation": "중간 grain과 다단계 계산 순서를 포함한 후속 계산 계약"
    }}
  ],
  "aggregation_policy": "preserve_common_grain 또는 aggregate_to_common_grain",
  "incremental_column": "... 또는 null",
  "load_strategy": "full_refresh 또는 incremental",
  "design_reasoning": "..."
}}
"""

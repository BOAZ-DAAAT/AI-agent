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
- grain을 반드시 명확히 정의
- 가능한 한 원본 데이터의 행 수준(base grain)을 유지
- 우선 조인, 정제, 표준화, 필수 파생 컬럼 추가로 해결
- 특정 질문의 최종 요약 결과 테이블처럼 과하게 집계하지 말 것
- key_columns, dimension_columns, measure_columns를 분리
- target_schema는 "{ALLOWED_MART_SCHEMA}" 로 고정
- incremental이 자연스러우면 incremental_column 제안
- 집계를 사용해야 한다면 왜 row-level mart가 부적절한지 aggregation_rationale과 design_reasoning에 설명
- 질문에 없는 정의를 과도하게 추가하지 말고 reasoning에 근거 설명
- 반드시 JSON만 출력

출력 형식:
{{
  "mart_name": "...",
  "target_schema": "{ALLOWED_MART_SCHEMA}",
  "grain": "...",
  "base_grain": "...",
  "source_tables": ["..."],
  "key_columns": ["..."],
  "measure_columns": ["..."],
  "dimension_columns": ["..."],
  "incremental_column": "... 또는 null",
  "load_strategy": "full_refresh 또는 incremental",
  "row_preserving_strategy": "원본 행 수준 유지 전략",
  "aggregation_policy": "prefer_row_preserving 또는 aggregate_if_justified",
  "aggregation_rationale": "... 또는 null",
  "design_reasoning": "..."
}}
"""

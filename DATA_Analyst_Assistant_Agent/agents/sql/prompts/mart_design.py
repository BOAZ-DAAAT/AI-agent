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
- key_columns, dimension_columns, measure_columns를 분리
- target_schema는 "{ALLOWED_MART_SCHEMA}" 로 고정
- incremental이 자연스러우면 incremental_column 제안
- 질문에 없는 정의를 과도하게 추가하지 말고 reasoning에 근거 설명
- 반드시 JSON만 출력

출력 형식:
{{
  "mart_name": "...",
  "target_schema": "{ALLOWED_MART_SCHEMA}",
  "grain": "...",
  "source_tables": ["..."],
  "key_columns": ["..."],
  "measure_columns": ["..."],
  "dimension_columns": ["..."],
  "incremental_column": "... 또는 null",
  "load_strategy": "full_refresh 또는 incremental",
  "design_reasoning": "..."
}}
"""

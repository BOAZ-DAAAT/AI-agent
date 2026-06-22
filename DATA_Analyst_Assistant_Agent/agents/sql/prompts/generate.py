"""SQL 생성(generate_sql) 프롬프트 — 마트/조회 두 가지."""

from __future__ import annotations

import json

from DATA_Analyst_Assistant_Agent.agents.sql._runtime import ALLOWED_MART_SCHEMA


def generate_mart_prompt(state, feedback: str) -> str:
    return f"""
너는 MySQL 데이터마트 생성 SQL 작성기다.

사용자 질문:
{state['user_question']}

질문 분석 결과:
{json.dumps(state['plan'], ensure_ascii=False, indent=2)}

마트 설계 결과:
{json.dumps(state.get('mart_design', {}), ensure_ascii=False, indent=2)}

스키마 JSON:
{state['schema_text']}

정합성 점검 JSON:
{state['integrity_text']}

이전 피드백:
{feedback if feedback else "없음"}

규칙:
- CREATE TABLE ... AS SELECT 또는 INSERT INTO ... SELECT 형태만 허용
- 타겟 스키마는 반드시 {ALLOWED_MART_SCHEMA}
- source는 실제 존재 테이블만 사용
- grain이 깨지지 않게 집계
- 모호한 기준은 reasoning에 명시
- precheck_sql에는 원천 데이터 건수/기간 확인용 SELECT
- postcheck_sql에는 생성 후 row_count / 중복 / null 점검용 SELECT
- DROP, ALTER, TRUNCATE 금지
- 반드시 JSON만 출력

출력 형식:
{{
  "sql": "...",
  "sql_type": "create_table_as 또는 insert_select",
  "target_table": "{ALLOWED_MART_SCHEMA}.xxx",
  "source_tables": ["..."],
  "columns_used": ["..."],
  "business_grain": "...",
  "precheck_sql": "SELECT ...",
  "postcheck_sql": "SELECT ...",
  "reasoning": "..."
}}
"""


def generate_query_prompt(state, feedback: str) -> str:
    return f"""
너는 MySQL 조회 SQL 작성기다.

사용자 질문:
{state['user_question']}

질문 분석 결과:
{json.dumps(state['plan'], ensure_ascii=False, indent=2)}

스키마 JSON:
{state['schema_text']}

정합성 점검 JSON:
{state['integrity_text']}

이전 피드백:
{feedback if feedback else "없음"}

규칙:
- MySQL SELECT SQL만 생성
- WITH 절 허용
- 질문에 없는 조건 임의 추가 금지
- 정합성 문제가 있는 컬럼/테이블 주의
- 반드시 JSON만 출력

출력 형식:
{{
  "sql": "SELECT ...",
  "sql_type": "select",
  "target_table": null,
  "source_tables": ["..."],
  "columns_used": ["..."],
  "business_grain": null,
  "precheck_sql": null,
  "postcheck_sql": null,
  "reasoning": "..."
}}
"""

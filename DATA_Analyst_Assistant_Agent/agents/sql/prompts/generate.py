"""SQL 생성(generate_sql) 프롬프트 — 마트/조회 두 가지."""

from __future__ import annotations

import json

from DATA_Analyst_Assistant_Agent.agents.sql._runtime import ALLOWED_MART_SCHEMA


def generate_mart_prompt(state, feedback: str) -> str:
    return f"""
너는 MySQL SQL 작성기다. 데이터마트 생성 SQL을 작성한다.

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
- 데이터마트는 최종 리포트용 요약 결과보다 재사용 가능한 기반 테이블이어야 함
- 가능한 한 원본 데이터의 행 수준 grain을 유지
- 우선 조인, 정제, 표준화, 필수 파생 컬럼 추가로 해결
- 집계는 꼭 필요한 경우에만 최소 수준으로 사용
- 집계를 사용했다면 왜 row-level mart가 부적절한지 reasoning에 명시
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
너는 MySQL SQL 작성기다. 조회 SQL을 작성한다.

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
- 좋은 simple 조회 SQL은 데이터마트처럼 원본 행 수준(row-preserving)을 최대한 유지한다
- 좋은 simple 조회 SQL은 필요한 테이블을 조인하고, 날짜/금액/상태 같은 분석용 컬럼을 정제·표준화·파생 컬럼으로 준비해 downstream 분석에 바로 쓸 수 있게 만든다
- 사용자가 명시적으로 합계/평균/건수/집계를 요구하지 않았다면, 날짜별 요약표보다 주문/아이템 등 상세 행을 유지한 분석용 테이블 형태가 더 좋은 답이다
- 추이 분석 질문이라도 simple 단계에서는 purchase_date, item_revenue 같은 파생 컬럼을 만든 상세 데이터셋이 좋은 기본 출력이다
- GROUP BY, HAVING, SUM, AVG, COUNT 같은 요약 집계는 사용자가 직접 집계를 요구했거나 row-level 결과가 질문에 답하기 어려울 때만 선택한다
- 집계를 선택했다면 reasoning에 왜 상세 row-preserving 조회보다 집계 결과가 더 적합한지 설명
- source_tables와 columns_used는 실제 SQL과 일치하게 정확히 채운다
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

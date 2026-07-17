"""SQL 생성(generate_sql) 프롬프트 — 마트/조회 두 가지."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql._runtime import ALLOWED_MART_SCHEMA


def generate_mart_prompt(generation_context_json: str) -> str:
    return f"""
너는 MySQL SQL 작성기다. 데이터마트 생성 SQL을 작성한다.

SQL 생성 컨텍스트(compact JSON):
{generation_context_json}

규칙:
- CREATE TABLE ... AS SELECT 형태만 허용
- 타겟 스키마는 반드시 {ALLOWED_MART_SCHEMA}
- source는 실제 존재 테이블만 사용
- 사용자 질문보다 확정된 mart_design을 우선 계약으로 사용
- mart_design의 target_table, grain_columns, source_grains, deduplication_keys,
  grain_strategy, column_plan, metric_support를 임의로 변경하지 말 것

- source 테이블에는 schema/database prefix를 절대 붙이지 말 것
- source 테이블은 customers, orders처럼 bare table name만 사용
- raw_data.*, source.*, public.*, olist.*, 세션 DB명 prefix는 금지
- schema prefix가 허용되는 것은 target_table의 {ALLOWED_MART_SCHEMA}.* 뿐

- 데이터마트는 최종 리포트용 요약 결과가 아니라 재사용 가능한 기반 테이블로 작성
- 기본적으로 가능한 한 원본 데이터의 행 수준 또는 mart_design에 선언된 공통 grain을 유지
- 우선 조인, 정제, 표준화, 중복 제거, 필수 파생 컬럼 추가로 해결
- preserve_common_grain이면 집계하지 않고 선언된 공통 grain을 보존
- aggregate_to_common_grain이면 mart_design에 선언된 집계 계약만 사용
- source_grains가 더 세밀한 원천은 deduplication_keys와
  각 column_plan.aggregation_method에 따라 공통 grain으로 집계하거나 중복 제거
- mart_design에 명시되지 않은 임의의 집계는 금지
- 집계를 사용했다면 row-level 또는 공통 grain 보존이 부적절한 이유를 reasoning에 명시

- mart_design.grain_columns가 최종 결과의 한 행을 유일하게 식별하도록 작성
- 최종 SELECT에는 column_plan의 output_column만 선언된 순서와 alias로 정확히 출력
- calculation_rule의 자연어 의미를 SQL로 구현하되 임의의 새 출력 컬럼을 추가하지 말 것

- metric_support의 downstream_calculation을 후속 단계에서 수행할 수 있도록
  required_mart_columns를 보존
- 질문 분석 결과의 required_columns와 business_keys는 원천 컬럼 참조와
  조인 조건을 결정하는 데 사용하되 mart_design 자체를 변경하지 말 것

- 재사용 가능한 원자적 파생값은 포함할 수 있음
- 비율, 순위, 최종 재구매 판정, 카테고리별 요약 지표 등
  최종 분석 또는 리포트 성격의 파생값은 생성하지 말 것
- 모호한 기준은 reasoning에 명시
- precheck_sql에는 원천 데이터 건수/기간 확인용 SELECT
- postcheck_sql은 타겟 마트에 대한 단일 SELECT이며 정확히 한 행을 반환
- postcheck_sql은 전체 행 수 AS row_count, grain_columns 기준 2행 이상인 grain 그룹 수 AS duplicate_grain_count, grain 컬럼 중 하나라도 NULL인 행 수 AS null_grain_count를 모두 제공
- DROP, ALTER, TRUNCATE 금지
- source_column_refs에는 실제 원천 table.column만 작성하고, 계산 alias나 최종 출력 alias는 넣지 말 것
- derived_columns에는 계산식으로 만든 alias만, output_columns에는 최종 SELECT에 노출되는 컬럼만 작성
- 반드시 JSON만 출력

출력 형식:
{{
  "sql": "...",
  "sql_type": "create_table_as",
  "target_table": "{ALLOWED_MART_SCHEMA}.xxx",
  "source_tables": ["..."],
  "source_column_refs": ["table.column"],
  "derived_columns": ["계산식 alias"],
  "output_columns": ["최종 노출 컬럼"],
  "business_grain": "mart_design.grain과 동일한 값",
  "precheck_sql": "SELECT ...",
  "postcheck_sql": "SELECT ...",
  "reasoning": "..."
}}
"""


def generate_query_prompt(generation_context_json: str) -> str:
    return f"""
너는 MySQL SQL 작성기다. 조회 SQL을 작성한다.

SQL 생성 컨텍스트(compact JSON):
{generation_context_json}

규칙:
 - MySQL SELECT SQL만 생성
 - WITH 절 허용
 - 필요하면 여러 개의 SELECT/WITH 문을 세미콜론으로 구분해 출력할 수 있다
 - 각 statement는 반드시 SELECT 또는 WITH 로 시작해야 한다
 - simple 경로에서는 INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/CREATE 금지
 - 이전에 특정 statement가 실패했다면 전체를 무작정 다시 쓰지 말고 실패 statement를 우선 수정하라
 - 질문에 없는 조건 임의 추가 금지
 - 질문 분석 결과의 required_columns를 SELECT 컬럼 선택의 우선 근거로 사용
 - 질문 분석 결과의 business_keys를 조인 조건과 식별자 선택의 우선 근거로 사용
 - 정합성 문제가 있는 컬럼/테이블 주의
 - source_column_refs에는 실제 원천 table.column만 작성하고, 계산 alias나 최종 출력 alias는 넣지 말 것
 - derived_columns에는 계산식으로 만든 alias만, output_columns에는 최종 SELECT에 노출되는 컬럼만 작성
 - 반드시 JSON만 출력

출력 형식:
{{
  "sql": "SELECT ...",
  "sql_type": "select",
  "target_table": null,
  "source_tables": ["..."],
  "source_column_refs": ["table.column"],
  "derived_columns": ["계산식 alias"],
  "output_columns": ["최종 노출 컬럼"],
  "business_grain": null,
  "precheck_sql": null,
  "postcheck_sql": null,
  "reasoning": "..."
}}
"""

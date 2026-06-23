from __future__ import annotations

import json
from typing import Any

from DATA_Analyst_Assistant_Agent.agents.sql.sql_agent.tool.runtime import MYSQL_DIALECT_NAME


def planner_prompt(state: dict[str, Any]) -> str:
    return f"""
너는 MySQL SQL 작성 전용 planner다.

사용자 질문:
{state['user_question']}

필요한 DB 스키마:
{state['schema_text']}

clarification 요청:
{state.get('clarification_request', '') or '없음'}

플랜 에이전트가 이 SQL 에이전트를 선택한 이유:
{state.get('planner_selection_reason', '') or '없음'}

규칙:
- 이 에이전트의 목적은 분석 리포트 작성이 아니라 적절한 SQL 작성이다.
- route_kind는 반드시 simple 또는 comprehensive 중 하나다.
- simple은 간단한 분석용 조회 SQL 경로다.
- comprehensive는 복잡한 분석을 위한 datamart 생성 경로다.
- selected_join_tables에는 실제로 조회/조인에 사용할 핵심 테이블을 넣는다.
- relevant_tables, candidate_tables는 반드시 실제 스키마에 있는 테이블만 사용한다.
- ambiguity_note에는 애매한 점만 간단히 적는다.
- 반드시 {MYSQL_DIALECT_NAME} 문법만 사용한다.
- SQLite/PostgreSQL 문법(JULIANDAY, STRFTIME, DATE_TRUNC, ILIKE, ::type 등)은 절대 사용하지 않는다.
- 날짜 차이는 MySQL 기준으로 DATEDIFF 또는 TIMESTAMPDIFF를 사용한다.
- reasoning은 반드시 한국어로 작성한다.
- 반드시 JSON만 출력한다.

출력 형식:
{{
  "original_question": "...",
  "route_kind": "simple 또는 comprehensive",
  "task_type": "query_answer 또는 data_mart_build",
  "requested_output": "sql_only / execute_and_answer / create_table",
  "target_metric": "...",
  "dimensions": ["..."],
  "filters": ["..."],
  "time_condition": "... 또는 null",
  "selected_join_tables": ["..."],
  "relevant_tables": ["..."],
  "candidate_tables": ["..."],
  "mart_name": "... 또는 null",
  "grain": "... 또는 null",
  "load_strategy": "... 또는 null",
  "ambiguity_note": "... 또는 null",
  "expected_result_shape": "single_scalar / grouped_aggregate / table_preview / datamart_creation",
  "required_columns": ["..."],
  "required_aggregations": ["AVG / COUNT / SUM 등"],
  "validation_contract": {{"expected_aliases": ["..."], "required_tables": ["..."]}},
  "reasoning": "한국어 근거"
}}
"""


def mart_design_prompt(state: dict[str, Any]) -> str:
    return f"""
너는 분석용 datamart 설계 보조자다.

사용자 질문:
{state['user_question']}

planner 결과:
{json.dumps(state['plan'], ensure_ascii=False, indent=2)}

스키마 JSON:
{state['schema_text']}

규칙:
- comprehensive 경로에서만 사용할 datamart 설계만 작성한다.
- source_tables는 planner의 selected_join_tables를 우선 사용한다.
- reasoning은 한국어로 작성한다.
- 반드시 JSON만 출력한다.
"""


def sql_generation_prompt(state: dict[str, Any], feedback: str) -> str:
    return f"""
너는 MySQL SQL 작성기다. 대상 DB는 {MYSQL_DIALECT_NAME} 이다.

사용자 질문:
{state['user_question']}

planner 결과:
{json.dumps(state['plan'], ensure_ascii=False, indent=2)}

플랜 에이전트가 이 SQL 에이전트를 선택한 이유:
{state.get('planner_selection_reason', '') or '없음'}

clarification 요청:
{state.get('clarification_request', '') or '없음'}

마트 설계 결과:
{json.dumps(state.get('mart_design', {}), ensure_ascii=False, indent=2)}

스키마 JSON:
{state['schema_text']}

정합성 점검 JSON:
{state['integrity_text']}

이전 피드백:
{feedback if feedback else '없음'}

규칙:
- route_kind=simple이면 간단한 분석용 조회 SQL만 생성한다.
- route_kind=comprehensive이면 datamart 생성 SQL만 생성한다.
- 조인 테이블은 planner의 selected_join_tables를 우선 사용한다.
- planner.validation_contract가 있으면 반드시 그 의도와 결과 형태를 만족하도록 SQL을 작성한다.
- 단일 집계 질문이면 aggregate 함수와 alias를 반드시 포함한다.
- 그룹 집계 질문이면 dimension 컬럼과 GROUP BY를 명시한다.
- comprehensive면 target_table과 postcheck_sql이 일관되게 연결되어야 한다.
- 반드시 {MYSQL_DIALECT_NAME} 문법만 사용한다.
- SQLite/PostgreSQL 문법(JULIANDAY, STRFTIME, DATE_TRUNC, ILIKE, ::type 등)은 절대 사용하지 않는다.
- 날짜 차이는 MySQL 기준으로 DATEDIFF 또는 TIMESTAMPDIFF를 사용한다.
- reasoning은 반드시 한국어로 작성한다.
- 반드시 JSON만 출력한다.

출력 형식:
{{
  "sql": "...",
  "sql_type": "select / create_table_as / insert_select",
  "target_table": null 또는 "analytics.xxx",
  "source_tables": ["..."],
  "columns_used": ["..."],
  "business_grain": null 또는 "...",
  "precheck_sql": null 또는 "SELECT ...",
  "postcheck_sql": null 또는 "SELECT ...",
  "reasoning": "한국어 근거"
}}
"""

"""검증(validate_sql_and_result) 프롬프트."""

from __future__ import annotations

import json

from DATA_Analyst_Assistant_Agent.agents.sql._runtime import format_result_rows


def validate_prompt(state) -> str:
    return f"""
너는 SQL/데이터마트 검증기다.

사용자 질문:
{state['user_question']}

질문 분석 결과:
{json.dumps(state['plan'], ensure_ascii=False, indent=2)}

마트 설계 결과:
{json.dumps(state.get('mart_design', {}), ensure_ascii=False, indent=2)}

정합성 점검 JSON:
{state['integrity_text']}

생성된 SQL:
{state['sql_draft']['sql']}

SQL 설명:
{state['sql_draft']['reasoning']}

사전 점검 결과:
{format_result_rows(state.get('precheck_result'))}

실행 결과:
{format_result_rows(state['sql_result'])}

사후 점검 결과:
{format_result_rows(state.get('postcheck_result'))}

행 수:
{state['row_count']}

검증 규칙:
1. task_type=query_answer 이면 질문 조건 충족 여부 검증
2. task_type=data_mart_build 이면 grain 적합성, 타겟 테이블 적절성, 재사용성 검증
3. 질문에 없는 조건 추가면 invalid
4. 정합성 문제 무시하면 invalid
5. 반드시 JSON만 출력

출력 형식:
{{
  "result": "valid" 또는 "invalid",
  "reason": "...",
  "feedback": "..."
}}
"""

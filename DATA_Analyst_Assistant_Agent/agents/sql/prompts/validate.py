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

마트 재사용성 사전 신호:
{json.dumps(state.get('mart_validation_signal', {}), ensure_ascii=False, indent=2)}

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
2. task_type=data_mart_build 이면 grain 적합성, 타겟 테이블 적절성, 재사용성, row-preserving 성격을 우선 검증
3. 데이터마트가 특정 질문에 대한 최종 요약 결과 테이블처럼 과하게 집계되면 invalid 또는 강한 수정 피드백
4. 집계를 사용했다면 row-level mart가 왜 부적절한지 정당화가 있는지 확인
5. 위 사전 신호는 참고용이며, 특정 키워드가 없다고 자동 invalid로 보지 말고 SQL의 실제 재사용성과 grain을 기준으로 판단
6. 질문에 없는 조건 추가면 invalid
7. 정합성 문제 무시하면 invalid
8. 반드시 JSON만 출력

출력 형식:
{{
  "result": "valid" 또는 "invalid",
  "reason": "...",
  "feedback": "..."
}}
"""

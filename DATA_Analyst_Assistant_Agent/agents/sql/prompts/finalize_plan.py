"""후보 상세 스키마를 이용한 최종 테이블 계획 프롬프트."""

from __future__ import annotations

import json


def finalize_table_plan_prompt(state) -> str:
    return f"""
너는 MySQL 물리 테이블 계획자다.

사용자 질문:
{state['user_question']}

정규화된 질문 계획:
{json.dumps(state.get('question_plan', {}), ensure_ascii=False, indent=2)}

후보 테이블 상세 스키마:
{state['schema_text']}

후보 테이블 정합성 정보:
{state['integrity_text']}

규칙:
- 제공된 후보 상세 스키마만 근거로 최종 테이블, 컬럼, 비즈니스 키를 결정
- selected_join_tables와 required_columns는 각각 최소 1개 이상 작성
- required_columns는 가능한 한 table.column 형식으로 작성
- business_keys는 테이블별 대표 식별자를 기록하고 적절한 키가 없으면 빈 객체 허용
- 후보 밖의 테이블을 탐색하거나 자동으로 추가하지 말 것
- 반드시 JSON object만 출력

출력 형식:
{{
  "selected_join_tables": ["실제 후보 테이블명"],
  "required_columns": ["table.column"],
  "business_keys": {{"table": "table.column"}},
  "reasoning": "최종 테이블·컬럼·키 선택 근거"
}}
"""

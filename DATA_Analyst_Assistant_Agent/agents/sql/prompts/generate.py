"""SQL 생성 프롬프트: 검증된 컨텍스트를 짧은 우선순위 계약으로 전달한다."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql._runtime import ALLOWED_MART_SCHEMA
from DATA_Analyst_Assistant_Agent.agents.sql.generation_context import (
    ComprehensiveSQLGenerationContext,
    SimpleSQLGenerationContext,
)


def _context_json(context: SimpleSQLGenerationContext | ComprehensiveSQLGenerationContext) -> str:
    return context.model_dump_json(exclude_none=True, by_alias=True)


def _integrity_rule(context: SimpleSQLGenerationContext | ComprehensiveSQLGenerationContext) -> str:
    if not context.integrity_failures:
        return ""
    return (
        "\n- 관련 무결성 실패만 팬아웃 방지·중복 제거·타입 변환에 반영한다. "
        "테이블/컬럼을 금지하거나 새 식별자를 만들지 않는다."
    )


def generate_mart_prompt(context: ComprehensiveSQLGenerationContext) -> str:
    """comprehensive route의 마트 생성 계약을 구성한다."""
    if not isinstance(context, ComprehensiveSQLGenerationContext):
        raise TypeError("마트 프롬프트에는 ComprehensiveSQLGenerationContext가 필요합니다")

    if context.aggregation_policy == "preserve_common_grain":
        aggregation_rule = "- preserve_common_grain: 집계 없이 final_grain을 보존한다."
    else:
        aggregation_rule = (
            "- aggregate_to_common_grain: column_plan.aggregation_method와 "
            "deduplication_keys로만 final_grain에 집계한다."
        )

    return f"""역할
MySQL 재사용 데이터마트 SQL을 작성한다.

계약 우선순위
1. mart_design(target_table, source_grains, final_grain, column_plan, metric_support, aggregation_policy)
2. schema와 selected_tables
3. user_question
4. previous_feedback

생성 컨텍스트
{_context_json(context)}

핵심 불변 조건
- comprehensive route이며 CREATE TABLE ... AS SELECT 한 문장만 생성한다.
- target은 {ALLOWED_MART_SCHEMA}.*만, source는 selected_tables의 bare table name만 사용한다.
- final_grain을 보존하고 grain_columns가 행을 식별하게 한다.
- 최종 컬럼은 column_plan.output_column의 순서·alias·계산 계약과 정확히 일치시킨다.
- column_plan에 선언된 분석 필수 파생변수는 비율이어도 calculation_rule의 분자·분모·연산 순서와 0/NULL 처리 규칙대로 SQL에서 생성한다.
- metric_support.required_mart_columns를 보존하며 임의 컬럼·집계·필터를 추가하지 않는다.
- source_column_refs는 실제 source table.column, derived_columns는 계산 alias, output_columns는 최종 컬럼만 기록한다.
- DROP, ALTER, TRUNCATE와 column_plan에 없는 최종 표시용 비율·순위·판정 지표를 생성하지 않는다.

조건부 규칙
{aggregation_rule}
- precheck_sql은 원천 건수/기간 SELECT로 작성한다.
- postcheck_sql은 target을 검사하는 한 행 SELECT로 row_count, duplicate_grain_count, null_grain_count를 반환한다.{_integrity_rule(context)}
- previous_feedback이 있으면 관련 실패만 수정한다.

SQLDraft 스키마에 맞춰 응답한다."""


def generate_query_prompt(context: SimpleSQLGenerationContext) -> str:
    """simple route의 조회 생성 계약을 구성한다."""
    if not isinstance(context, SimpleSQLGenerationContext):
        raise TypeError("조회 프롬프트에는 SimpleSQLGenerationContext가 필요합니다")

    return f"""역할
MySQL 조회 SQL을 작성한다.

계약 우선순위
1. schema와 selected_tables
2. user_question
3. previous_feedback

생성 컨텍스트
{_context_json(context)}

핵심 불변 조건
- simple route이며 SELECT 또는 WITH statement만 생성한다. 여러 statement는 세미콜론으로 구분할 수 있다.
- source는 selected_tables에 있고 schema에 존재하는 bare table/column만 사용한다.
- required_columns는 출력, business_keys는 조인·식별의 우선 근거로 사용한다.
- 질문에 없는 컬럼·집계·필터를 임의로 추가하지 않는다.
- INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE를 사용하지 않는다.
- source_column_refs는 실제 source table.column, derived_columns는 계산 alias, output_columns는 최종 컬럼만 기록한다.{_integrity_rule(context)}

조건부 규칙
- previous_feedback이 있으면 실패 statement를 우선 수정한다.

SQLDraft 스키마에 맞춰 응답한다."""

"""SQL 에이전트의 상태/IO 모델.

기존 `sql_agent/sql_agent.py` 의 Pydantic 모델과 LangGraph AgentState 를 분리한 것.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator


# -----------------------------
# Pydantic Models
# -----------------------------
class QuestionPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    route_kind: str = Field(description="simple / comprehensive")
    question_type: str = Field(min_length=1, description="aggregation/comparison/ranking/filter/detail/trend/identification/mart_build")
    target_metrics: List[str] = Field(description="핵심 지표 목록")
    analysis_entities: List[str] = Field(description="분석 주체 목록")
    dimensions: List[str] = Field(description="그룹 기준")
    filters: List[str] = Field(description="필터와 시간 조건")
    candidate_tables: List[str] = Field(description="상세 검토 후보 테이블")
    required_aggregations: List[str] = Field(description="필수 집계 함수")
    reasoning: str = Field(min_length=1, description="후보 선정과 경로 판단 근거")

    @field_validator("question_type", "reasoning")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("빈 문자열은 허용되지 않습니다")
        return value


class FinalTablePlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    selected_join_tables: List[str] = Field(min_length=1, description="최종 조회 및 조인 테이블")
    required_columns: List[str] = Field(min_length=1, description="최종 필수 컬럼")
    business_keys: Dict[str, str] = Field(description="테이블별 비즈니스 키")
    reasoning: str = Field(min_length=1, description="물리 테이블 계획 근거")

    @field_validator("reasoning")
    @classmethod
    def reject_blank_reasoning(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("빈 문자열은 허용되지 않습니다")
        return value


class MartDesign(BaseModel):
    mart_name: str
    target_schema: str
    grain: str
    base_grain: str = Field(default="원본 entity/event grain 유지", description="가능하면 유지할 기본 행 수준 grain")
    source_tables: List[str] = Field(default_factory=list)
    key_columns: List[str] = Field(default_factory=list)
    measure_columns: List[str] = Field(default_factory=list)
    dimension_columns: List[str] = Field(default_factory=list)
    incremental_column: Optional[str] = None
    load_strategy: str = "full_refresh"
    row_preserving_strategy: str = Field(default="원본 행 수준 유지 우선", description="row-preserving 설계 전략")
    aggregation_policy: str = Field(default="prefer_row_preserving", description="prefer_row_preserving / aggregate_if_justified")
    aggregation_rationale: Optional[str] = Field(default=None, description="집계를 사용한 경우 정당화 근거")
    design_reasoning: str


class SQLDraft(BaseModel):
    sql: str
    sql_type: str = Field(description="select / create_table_as")
    target_table: Optional[str] = None
    source_tables: List[str] = Field(default_factory=list)
    source_column_refs: List[str] = Field(default_factory=list)
    derived_columns: List[str] = Field(default_factory=list)
    output_columns: List[str] = Field(default_factory=list)
    business_grain: Optional[str] = None
    precheck_sql: Optional[str] = None
    postcheck_sql: Optional[str] = None
    reasoning: str


class ValidationResult(BaseModel):
    result: str
    reason: str
    feedback: str
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    retry_hint: Dict[str, Any] = Field(default_factory=dict)


class AgentState(TypedDict):
    user_question: str
    required_db_schema: str
    clarification_request: str
    planner_selection_reason: str
    schema_text: str
    integrity_text: str
    integrity_dataset_name: str
    integrity_preplan: Dict[str, Any]
    integrity_refresh: Dict[str, Any]
    schema_refresh: Dict[str, Any]
    question_plan: Dict[str, Any]
    final_table_plan: Dict[str, Any]
    planning_stages: Dict[str, Any]

    plan: Dict[str, Any]
    mart_design: Dict[str, Any]
    sql_draft: Dict[str, Any]

    sql_result: Any
    statement_results: List[Dict[str, Any]]
    row_count: int

    precheck_result: Any
    postcheck_result: Any
    mart_quality_result: Dict[str, Any]

    validation: Dict[str, Any]
    validation_findings: List[Dict[str, Any]]
    retry_hint: Dict[str, Any]
    validation_summary: Dict[str, Any]
    retry_count: int
    max_retries: int
    feedback: str
    error: str
    generation_source: str
    generation_failure_reason: str
    failed_statement_index: Optional[int]
    failed_statement_sql: str
    final_answer: str

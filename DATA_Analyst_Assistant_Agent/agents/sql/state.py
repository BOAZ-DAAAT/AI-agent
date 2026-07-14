"""SQL 에이전트의 상태/IO 모델.

기존 `sql_agent/sql_agent.py` 의 Pydantic 모델과 LangGraph AgentState 를 분리한 것.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


# -----------------------------
# Pydantic Models
# -----------------------------
class QuestionPlan(BaseModel):
    original_question: str = Field(description="사용자 원문 질문")
    route_kind: str = Field(description="simple / comprehensive")
    question_type: str = Field(description="aggregation/comparison/ranking/filter/detail/trend/identification/mart_build")
    task_type: str = Field(description="query_answer 또는 data_mart_build")
    requested_output: str = Field(description="sql_only / execute_and_answer / create_table")
    target_metric: str = Field(description="핵심 지표")
    dimensions: List[str] = Field(default_factory=list, description="그룹 기준")
    filters: List[str] = Field(default_factory=list, description="필터 조건")
    time_condition: Optional[str] = Field(default=None, description="시간 조건")
    selected_join_tables: List[str] = Field(default_factory=list, description="조인 또는 조회 대상 테이블")
    relevant_tables: List[str] = Field(default_factory=list, description="관련 테이블")
    candidate_tables: List[str] = Field(default_factory=list, description="검토 후보 테이블")
    mart_name: Optional[str] = Field(default=None, description="생성 대상 마트명")
    grain: Optional[str] = Field(default=None, description="마트 grain")
    load_strategy: Optional[str] = Field(default=None, description="full_refresh / incremental")
    ambiguity_note: Optional[str] = Field(default=None, description="애매한 표현")
    expected_result_shape: str = Field(description="table_preview / datamart_creation")
    required_columns: List[str] = Field(default_factory=list, description="반드시 필요하다고 판단한 컬럼")
    required_aggregations: List[str] = Field(default_factory=list, description="필수 집계 함수")
    validation_contract: Dict[str, Any] = Field(default_factory=dict, description="validation용 구조화 계약")
    reasoning: str = Field(default="", description="planner 근거")


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

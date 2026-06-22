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
    question_type: str = Field(description="aggregation/comparison/ranking/filter/detail/trend/identification/mart_build")
    task_type: str = Field(description="query_answer 또는 data_mart_build")
    requested_output: str = Field(description="sql_only / execute_and_answer / create_table")
    target_metric: str = Field(description="핵심 지표")
    dimensions: List[str] = Field(default_factory=list, description="그룹 기준")
    filters: List[str] = Field(default_factory=list, description="필터 조건")
    time_condition: Optional[str] = Field(default=None, description="시간 조건")
    relevant_tables: List[str] = Field(default_factory=list, description="관련 테이블")
    mart_name: Optional[str] = Field(default=None, description="생성 대상 마트명")
    grain: Optional[str] = Field(default=None, description="마트 grain")
    load_strategy: Optional[str] = Field(default=None, description="full_refresh / incremental")
    ambiguity_note: Optional[str] = Field(default=None, description="애매한 표현")


class MartDesign(BaseModel):
    mart_name: str
    target_schema: str
    grain: str
    source_tables: List[str] = Field(default_factory=list)
    key_columns: List[str] = Field(default_factory=list)
    measure_columns: List[str] = Field(default_factory=list)
    dimension_columns: List[str] = Field(default_factory=list)
    incremental_column: Optional[str] = None
    load_strategy: str = "full_refresh"
    design_reasoning: str


class SQLDraft(BaseModel):
    sql: str
    sql_type: str = Field(description="select / create_table_as / insert_select")
    target_table: Optional[str] = None
    source_tables: List[str] = Field(default_factory=list)
    columns_used: List[str] = Field(default_factory=list)
    business_grain: Optional[str] = None
    precheck_sql: Optional[str] = None
    postcheck_sql: Optional[str] = None
    reasoning: str


class ValidationResult(BaseModel):
    result: str
    reason: str
    feedback: str


class AgentState(TypedDict):
    user_question: str
    schema_text: str
    integrity_text: str

    plan: Dict[str, Any]
    mart_design: Dict[str, Any]
    sql_draft: Dict[str, Any]

    sql_result: Any
    row_count: int

    precheck_result: Any
    postcheck_result: Any
    mart_quality_result: Dict[str, Any]

    validation: Dict[str, Any]
    retry_count: int
    max_retries: int
    feedback: str
    error: str
    final_answer: str

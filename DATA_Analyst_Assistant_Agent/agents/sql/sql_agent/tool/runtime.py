from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from DATA_Analyst_Assistant_Agent.agents.sql.db.db_connect import get_db_engine
from DATA_Analyst_Assistant_Agent.llm import get_chat_model

load_dotenv()

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 2))
ALLOWED_MART_SCHEMA = os.getenv("ALLOWED_MART_SCHEMA", "analytics")
ALLOW_MART_WRITE = os.getenv("ALLOW_MART_WRITE", "true").lower() == "true"
MYSQL_DIALECT_NAME = "MySQL 8.x"


class QuestionPlan(BaseModel):
    original_question: str = Field(description="사용자 원문 질문")
    route_kind: str = Field(description="simple 또는 comprehensive")
    task_type: str = Field(description="query_answer 또는 data_mart_build")
    requested_output: str = Field(description="sql_only / execute_and_answer / create_table")
    target_metric: str = Field(description="핵심 지표")
    dimensions: List[str] = Field(default_factory=list, description="그룹 기준")
    filters: List[str] = Field(default_factory=list, description="필터 조건")
    time_condition: Optional[str] = Field(default=None, description="시간 조건")
    selected_join_tables: List[str] = Field(default_factory=list, description="조인 또는 조회 대상 테이블")
    relevant_tables: List[str] = Field(default_factory=list, description="관련 테이블")
    candidate_tables: List[str] = Field(default_factory=list, description="검토한 후보 테이블")
    mart_name: Optional[str] = Field(default=None, description="생성 대상 마트명")
    grain: Optional[str] = Field(default=None, description="마트 grain")
    load_strategy: Optional[str] = Field(default=None, description="full_refresh / incremental")
    ambiguity_note: Optional[str] = Field(default=None, description="애매한 표현")
    expected_result_shape: str = Field(default="table_preview", description="single_scalar / grouped_aggregate / table_preview / datamart_creation")
    required_columns: List[str] = Field(default_factory=list, description="반드시 필요하다고 판단한 컬럼")
    required_aggregations: List[str] = Field(default_factory=list, description="필수 집계 함수")
    validation_contract: Dict[str, Any] = Field(default_factory=dict, description="validation용 구조화 계약")
    reasoning: str = Field(default="", description="한국어 planner 근거")


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
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    retry_hint: Dict[str, Any] = Field(default_factory=dict)


class AgentState(TypedDict):
    user_question: str
    required_db_schema: str
    clarification_request: str
    planner_selection_reason: str
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
    validation_findings: List[Dict[str, Any]]
    retry_hint: Dict[str, Any]
    validation_summary: Dict[str, Any]
    retry_count: int
    max_retries: int
    feedback: str
    error: str
    final_answer: str


engine = get_db_engine()
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = get_chat_model(temperature=0)
    return _llm

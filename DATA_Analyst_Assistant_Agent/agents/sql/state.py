"""SQL 에이전트의 상태/IO 모델.

기존 `sql_agent/sql_agent.py` 의 Pydantic 모델과 LangGraph AgentState 를 분리한 것.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator


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


class MartColumnPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_column: str = Field(min_length=1)
    role: Literal["dimension", "measure", "attribute"]
    source_columns: List[str] = Field(min_length=1)
    calculation_type: Literal["passthrough", "derived"]
    calculation_rule: str = Field(min_length=1, description="SQL이 아닌 자연어 계산 의미")
    aggregation_method: Literal["none", "SUM", "COUNT", "COUNT_DISTINCT", "MIN", "MAX", "AVG", "DEDUPLICATE"]
    inclusion_reason: str = Field(min_length=1)

    @field_validator("output_column", "calculation_rule", "inclusion_reason")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("빈 문자열은 허용되지 않습니다")
        return value.strip()

    @field_validator("source_columns")
    @classmethod
    def validate_source_columns(cls, value: List[str]) -> List[str]:
        normalized = [str(item).strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("source_columns에는 빈 컬럼을 넣을 수 없습니다")
        if len(set(normalized)) != len(normalized):
            raise ValueError("source_columns는 중복될 수 없습니다")
        return normalized


class MetricSupport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_name: str = Field(min_length=1)
    calculation_grain: List[str]
    required_mart_columns: List[str] = Field(min_length=1)
    downstream_calculation: str = Field(min_length=1)

    @field_validator("metric_name", "downstream_calculation")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("빈 문자열은 허용되지 않습니다")
        return value.strip()

    @field_validator("calculation_grain", "required_mart_columns")
    @classmethod
    def validate_column_references(cls, value: List[str]) -> List[str]:
        normalized = [str(item).strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("컬럼 참조에는 빈 값을 넣을 수 없습니다")
        if len(set(normalized)) != len(normalized):
            raise ValueError("컬럼 참조는 중복될 수 없습니다")
        return normalized


class MartDesign(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mart_name: str
    target_schema: str
    grain: str = Field(min_length=1, description="하위 Agent에 전달할 공통 분석 grain 설명")
    grain_columns: List[str] = Field(min_length=1)
    source_grains: Dict[str, List[str]] = Field(min_length=1)
    deduplication_keys: List[str] = Field(min_length=1)
    column_plan: List[MartColumnPlan] = Field(min_length=1)
    metric_support: List[MetricSupport] = Field(min_length=1)
    aggregation_policy: Literal["preserve_common_grain", "aggregate_to_common_grain"]
    source_tables: List[str] = Field(min_length=1)
    incremental_column: Optional[str] = None
    load_strategy: str = "full_refresh"
    design_reasoning: str = Field(min_length=1)

    @computed_field(return_type=List[str])
    @property
    def key_columns(self) -> List[str]:
        return list(self.grain_columns)

    @computed_field(return_type=List[str])
    @property
    def dimension_columns(self) -> List[str]:
        return [item.output_column for item in self.column_plan if item.role == "dimension"]

    @computed_field(return_type=List[str])
    @property
    def measure_columns(self) -> List[str]:
        return [item.output_column for item in self.column_plan if item.role == "measure"]

    @model_validator(mode="after")
    def validate_design_contract(self) -> "MartDesign":
        output_columns = [item.output_column for item in self.column_plan]
        if len(set(output_columns)) != len(output_columns):
            raise ValueError("column_plan의 output_column은 중복될 수 없습니다")
        if len(set(self.grain_columns)) != len(self.grain_columns):
            raise ValueError("grain_columns는 중복될 수 없습니다")
        if self.deduplication_keys != self.grain_columns:
            raise ValueError("deduplication_keys는 grain_columns와 순서까지 동일해야 합니다")
        missing_grain = [column for column in self.grain_columns if column not in output_columns]
        if missing_grain:
            raise ValueError(f"grain_columns가 column_plan에 없습니다: {missing_grain}")
        if len(set(self.source_tables)) != len(self.source_tables):
            raise ValueError("source_tables는 중복될 수 없습니다")
        if set(self.source_grains) != set(self.source_tables):
            raise ValueError("source_grains는 source_tables의 모든 테이블과 정확히 일치해야 합니다")
        for table, columns in self.source_grains.items():
            if not columns or any(not str(column).strip() for column in columns):
                raise ValueError(f"{table}의 source grain 키가 비어 있습니다")
        has_aggregation = any(item.aggregation_method != "none" for item in self.column_plan)
        if self.aggregation_policy == "preserve_common_grain" and has_aggregation:
            raise ValueError("preserve_common_grain에서는 모든 aggregation_method가 none이어야 합니다")
        available = set(output_columns)
        metric_names = [item.metric_name for item in self.metric_support]
        if len(set(metric_names)) != len(metric_names):
            raise ValueError("metric_support의 metric_name은 중복될 수 없습니다")
        for metric in self.metric_support:
            unknown = (set(metric.calculation_grain) | set(metric.required_mart_columns)) - available
            if unknown:
                raise ValueError(f"metric_support가 미등록 마트 컬럼을 참조합니다: {sorted(unknown)}")
        return self


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
    previous_sql_draft: Dict[str, Any]

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
    failed_sql_component: Optional[Literal["precheck", "main", "postcheck"]]
    execution_error_info: Dict[str, Any]
    final_answer: str

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class AnalysisKind(str, Enum):
    descriptive = "descriptive"
    group_comparison = "group_comparison"
    correlation = "correlation"
    trend = "trend"
    regression = "regression"
    classification = "classification"
    anomaly_detection = "anomaly_detection"
    time_series = "time_series"
    cohort = "cohort"
    funnel = "funnel"
    segmentation = "segmentation"
    customer_value = "customer_value"
    contribution = "contribution"
    survival = "survival"
    text = "text"
    scenario = "scenario"
    journey = "journey"
    mix_shift = "mix_shift"
    mmm = "mmm"
    clv = "clv"
    geospatial = "geospatial"
    optimization = "optimization"


class AnalysisContext(BaseModel):
    """Bounded context passed to the planner instead of raw orchestration state."""

    user_question: str
    goal: str
    route_kind: str
    question_type: str | None = None
    metric_hint: str | None = None
    dimension_hint: str | None = None
    row_count: int = 0
    columns: list[str] = Field(default_factory=list)
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    temporal_columns: list[str] = Field(default_factory=list)
    column_profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)
    eda_quality_statuses: list[str] = Field(default_factory=list)
    eda_key_issues: list[str] = Field(default_factory=list)
    source_artifact_ids: list[str] = Field(default_factory=list)


class AnalysisExecutionPlan(BaseModel):
    """Structured planner output consumed by deterministic analysis tools."""

    objective: str
    question_type: str = "descriptive"
    analysis_kind: AnalysisKind = AnalysisKind.descriptive
    analysis_subtype: str = "descriptive_summary"
    tool_names: list[str] = Field(default_factory=list)
    metric: str | None = None
    dimension: str | None = None
    time_column: str | None = None
    feature_columns: list[str] = Field(default_factory=list)
    tool_parameters: dict[str, dict[str, Any]] = Field(default_factory=dict)
    confidence_requirements: list[str] = Field(default_factory=list)
    requires_human_review: bool = False
    review_reason: str = ""
    planner_mode: Literal["deterministic", "llm"] = "deterministic"


class AnalysisEvidence(BaseModel):
    tool_name: str
    method: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    statistics: dict[str, Any] = Field(default_factory=dict)
    finding: str
    caveats: list[str] = Field(default_factory=list)


class HypothesisSummary(BaseModel):
    null_hypothesis: str
    alternative_hypothesis: str
    decision: Literal["supported", "not_supported", "inconclusive", "not_tested"] = "not_tested"
    rationale: str


class HumanReview(BaseModel):
    required: bool = False
    reason: str = ""
    allowed_decisions: list[Literal["approve", "edit", "reject"]] = Field(
        default_factory=lambda: ["approve", "edit", "reject"]
    )


class AnalysisResult(BaseModel):
    """Stable structured output for validation, visualization, and reporting."""

    run_id: str
    goal: str
    plan: AnalysisExecutionPlan
    method_summary: str
    key_findings: list[str] = Field(default_factory=list)
    evidence: list[AnalysisEvidence] = Field(default_factory=list)
    hypotheses: list[HypothesisSummary] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    source_artifacts: dict[str, list[str]] = Field(default_factory=dict)
    data_quality_notes: list[str] = Field(default_factory=list)
    eda_profile_summaries: list[dict[str, Any]] = Field(default_factory=list)
    human_review: HumanReview = Field(default_factory=HumanReview)

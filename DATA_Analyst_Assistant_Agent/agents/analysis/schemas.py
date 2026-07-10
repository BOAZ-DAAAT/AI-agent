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
    general_task = "general_task"


TimeGrain = Literal["D", "W", "M", "Q", "Y"]


class AnalysisIntent(BaseModel):
    """Output of the `classify` node.

    Replaces the fixed question_type catalog with free-form intent plus a
    deterministic time grain. The LLM fills the intent fields; the node fills
    `time_grain`/`time_span_days` deterministically from the actual dataframe.
    Domain is a free-form label used only to pick a domain prompt template — it
    is NOT a hardcoded enum and is never used for branching logic.
    """

    objective: str
    analysis_focus: list[str] = Field(default_factory=list)
    domain: str = "general"
    metric_hints: list[str] = Field(default_factory=list)
    dimension_hints: list[str] = Field(default_factory=list)
    entity_hints: list[str] = Field(default_factory=list)
    time_column: str | None = None
    is_time_based: bool = False
    time_grain: TimeGrain | None = None
    time_span_days: int | None = None
    requires_human_review: bool = False
    review_reason: str = ""
    notes: str = ""


class GeneratedAnalysisCode(BaseModel):
    """Output of the `generate` node.

    Structured (pydantic-forced) so we never parse markdown fences by hand.
    `imports` is separated because import hallucination is the most common
    codegen failure and is checked in its own node. `code` must set a variable
    named `result` to a dict with summary/findings/statistics/limitations.
    """

    rationale: str
    imports: str = ""
    code: str


class CodeCritique(BaseModel):
    """Output of the `critic` node (method-validity only).

    Judges whether the generated code applies a statistically sound method to
    the data (right test, right grain, no leakage). It does NOT judge fit to
    user intent — that is the supervisor's responsibility. On `fail`, `feedback`
    is fed back into the next generate attempt.
    """

    verdict: Literal["pass", "fail"] = "pass"
    method_issues: list[str] = Field(default_factory=list)
    feedback: str = ""


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
    last_failure: dict[str, str] | None = None


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
    status: Literal["success", "failed"] = "success"
    method: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    statistics: dict[str, Any] = Field(default_factory=dict)
    finding: str
    caveats: list[str] = Field(default_factory=list)


class VisualEvidence(BaseModel):
    chart_artifact_id: str | None = None
    filename: str | None = None
    chart_type: str | None = None
    related_block: str | None = None
    related_keys: list[str] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)
    status: Literal["read_success", "not_available", "fetch_failed", "reader_failed", "reader_unavailable"] = "not_available"
    multimodal_summary: str = ""
    cautions: list[str] = Field(default_factory=list)


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


class AnswerCoverage(BaseModel):
    """Structured evidence for supervisor-level answer adequacy validation."""

    requested_metrics: list[str] = Field(default_factory=list)
    used_metrics: list[str] = Field(default_factory=list)
    requested_dimensions: list[str] = Field(default_factory=list)
    used_dimensions: list[str] = Field(default_factory=list)
    requested_time_column: str | None = None
    used_time_column: str | None = None
    requested_time_grain: TimeGrain | None = None
    used_time_grain: TimeGrain | None = None
    coverage_status: Literal["full", "partial", "missing"] = "full"
    missing_requirements: list[str] = Field(default_factory=list)


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
    chart_requests: list[dict[str, Any]] = Field(default_factory=list)
    visual_evidence: list[VisualEvidence] = Field(default_factory=list)
    chart_status: str = "not_needed"
    human_review: HumanReview = Field(default_factory=HumanReview)
    answer_coverage: AnswerCoverage = Field(default_factory=AnswerCoverage)

    # Codegen-path additions (optional; empty on the legacy tool-dispatch path).
    intent: AnalysisIntent | None = None
    generated_code: str = ""
    code_critique: CodeCritique | None = None
    codegen_attempts: int = 0

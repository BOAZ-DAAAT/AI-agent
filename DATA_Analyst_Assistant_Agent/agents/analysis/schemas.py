from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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

    verdict: Literal["pass", "review_required", "fail"] = "pass"
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
    eda_candidate_insights: list[str] = Field(default_factory=list)
    eda_candidate_hypotheses: list[str] = Field(default_factory=list)
    # 상류 SQL 원천 테이블의 GE 정합성 이슈(스코핑+fail_only). 코드생성/검증이 해석 한계로 참고(#130).
    known_data_quality_issues: list[str] = Field(default_factory=list)
    analysis_data_contract: dict[str, Any] = Field(default_factory=dict)
    mart_columns: list[dict[str, Any]] = Field(default_factory=list)
    contract_issues: list[str] = Field(default_factory=list)
    source_artifact_ids: list[str] = Field(default_factory=list)
    last_failure: dict[str, str] | None = None
    review_request: "ReviewRequest | None" = None
    selection_response: "AnalysisSelectionResponse | None" = None


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
    status: Literal["success", "review_required", "failed"] = "success"
    method: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    statistics: dict[str, Any] = Field(default_factory=dict)
    finding: str
    caveats: list[str] = Field(default_factory=list)


class HypothesisTestResult(BaseModel):
    hypothesis: str
    test_name: str
    null_hypothesis: str = ""
    alternative_hypothesis: str = ""
    statistic: float | None = None
    p_value: float | None = None
    effect_size: float | None = None
    n: int | None = None
    decision: Literal["supported", "inconclusive", "not_supported"] = "inconclusive"
    caveats: list[str] = Field(default_factory=list)


class EvidenceTable(BaseModel):
    title: str
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)


class MethodDecision(BaseModel):
    """Why the analysis selected one method rather than an alternative."""

    selected_method: str
    rationale: str
    assumptions_checked: list[str] = Field(default_factory=list)
    fallbacks_considered: list[str] = Field(default_factory=list)


class ReviewOption(BaseModel):
    """One actionable, mutually exclusive analysis path for human selection."""

    id: str
    label: str
    method: str
    assumptions: list[str] = Field(default_factory=list)
    advantages: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    impact: str
    recommended: bool = False

    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class ReviewRequest(BaseModel):
    decision_type: str = ""
    question: str = ""
    proposal: str = ""
    rationale: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    options: list[ReviewOption] = Field(default_factory=list)
    recommended_option_id: str = ""
    allow_free_text: bool = True
    free_text_prompt: str = "Provide a different analysis constraint or preference."
    impact_if_approved: str = ""
    requires_followup_analysis: bool = True

    @model_validator(mode="after")
    def validate_review_contract(self) -> "ReviewRequest":
        if not self.question.strip() or not self.proposal.strip():
            raise ValueError("review_request에는 question과 proposal이 필요합니다.")
        if len(self.options) < 2:
            raise ValueError("review_request에는 최소 두 개의 option이 필요합니다.")
        option_ids = [option.id.strip() for option in self.options]
        if any(not option_id for option_id in option_ids) or len(option_ids) != len(set(option_ids)):
            raise ValueError("review_request option ID는 비어 있지 않고 고유해야 합니다.")
        if self.recommended_option_id not in option_ids:
            raise ValueError("recommended_option_id는 review option을 참조해야 합니다.")
        recommended_ids = [option.id for option in self.options if option.recommended]
        if recommended_ids != [self.recommended_option_id]:
            raise ValueError("추천 option은 recommended_option_id와 일치하는 하나여야 합니다.")
        if any(
            not option.label.strip() or not option.method.strip() or not option.impact.strip()
            for option in self.options
        ):
            raise ValueError("각 review option에는 label, method, impact가 필요합니다.")
        if self.requires_followup_analysis:
            return self
        analyzed = self.evidence.get("analyzed_option_ids")
        if not isinstance(analyzed, list) or not all(isinstance(item, str) for item in analyzed):
            raise ValueError(
                "requires_followup_analysis=false이면 evidence.analyzed_option_ids가 필요합니다."
            )
        if len(analyzed) != len(set(analyzed)) or set(analyzed) != set(option_ids):
            raise ValueError(
                "evidence.analyzed_option_ids는 모든 option ID를 중복 없이 정확히 포함해야 합니다."
            )
        return self


class AnalysisSelectionResponse(BaseModel):
    """Future resume payload accepted by Analysis when a review request is answered."""

    model_config = ConfigDict(extra="forbid")

    selected_option_id: str | None = None
    free_text: str | None = None

    @field_validator("selected_option_id", "free_text", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        return value.strip()

    @model_validator(mode="after")
    def require_exactly_one_response(self) -> "AnalysisSelectionResponse":
        has_option = bool((self.selected_option_id or "").strip())
        has_free_text = bool((self.free_text or "").strip())
        if has_option == has_free_text:
            raise ValueError("provide exactly one of selected_option_id or free_text")
        return self

    def constraint_text(self) -> str:
        if self.free_text and self.free_text.strip():
            return self.free_text.strip()
        if self.selected_option_id and self.selected_option_id.strip():
            return f"Selected analysis option: {self.selected_option_id.strip()}"
        return ""


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
    status: Literal["success", "review_required", "failed"] = "success"
    title: str = ""
    executive_summary: str = ""
    hypothesis_tests: list[HypothesisTestResult] = Field(default_factory=list)
    evidence_tables: list[EvidenceTable] = Field(default_factory=list)
    interpretation: list[str] = Field(default_factory=list)
    review_request: ReviewRequest | None = None
    selection_response: AnalysisSelectionResponse | None = None
    method_decision: MethodDecision | None = None
    method_notes: list[str] = Field(default_factory=list)
    debug_artifact_id: str | None = None

    # Codegen-path additions (optional; empty on the legacy tool-dispatch path).
    intent: AnalysisIntent | None = None
    generated_code: str = ""
    code_critique: CodeCritique | None = None
    codegen_attempts: int = 0
    # Per-attempt failure record (stage: execute/result_contract/critic + raw
    # error/code), debug-only. Lets a repeated-failure diagnosis see which
    # stage actually rejected the code instead of guessing from terminal_reason.
    error_history: list[dict[str, Any]] = Field(default_factory=list)

from __future__ import annotations

from pydantic import Field

from data_agent_backend.models.common import BackendModel, JsonDict


class SQLAgentResult(BackendModel):
    query: str | None = None
    target_table: str | None = None
    mart_design: JsonDict = Field(default_factory=dict)
    artifact_refs: list[JsonDict] = Field(default_factory=list)


class EDAAgentResult(BackendModel):
    final_summary: str | None = None
    stats: JsonDict = Field(default_factory=dict)
    hypotheses: list[JsonDict] = Field(default_factory=list)
    chart_requests: list[JsonDict] = Field(default_factory=list)
    chart_refs: list[JsonDict] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class AnalysisAgentResult(BackendModel):
    hypothesis_results: list[JsonDict] = Field(default_factory=list)
    model_results: list[JsonDict] = Field(default_factory=list)
    interpretations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ReportSection(BackendModel):
    title: str
    content: str
    evidence_refs: list[JsonDict] = Field(default_factory=list)


class ReportAgentInput(BackendModel):
    user_question: str
    run_id: str
    question_type: str | None = None
    supervisor_reason: str | None = None
    sql_result: SQLAgentResult | None = None
    eda_result: EDAAgentResult | None = None
    analysis_result: AnalysisAgentResult | None = None
    chart_refs: list[JsonDict] = Field(default_factory=list)
    artifact_refs: list[JsonDict] = Field(default_factory=list)
    language: str = "ko"


class ReportAgentOutput(BackendModel):
    title: str
    final_summary: str
    key_findings: list[str] = Field(default_factory=list)
    sections: list[ReportSection] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_questions: list[str] = Field(default_factory=list)
    chart_refs: list[JsonDict] = Field(default_factory=list)
    artifact_refs: list[JsonDict] = Field(default_factory=list)
    report_markdown: str
    report_artifact_id: str | None = None

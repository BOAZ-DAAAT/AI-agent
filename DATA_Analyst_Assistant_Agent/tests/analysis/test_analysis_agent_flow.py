"""AnalysisAgent end-to-end registration test on the codegen-first flow."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

from data_agent_backend.config import BackendConfig
from data_agent_backend.models.artifacts import ArtifactType
from DATA_Analyst_Assistant_Agent.agents.analysis import AnalysisAgent, AnalysisResult
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisIntent,
    CodeCritique,
    GeneratedAnalysisCode,
)
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.shared.contracts import AnalysisPlan, OrchestrationState
from DATA_Analyst_Assistant_Agent.shared.contracts import AgentStatus


class _Structured:
    def __init__(self, queue: list[object]) -> None:
        self._queue = queue

    def invoke(self, _messages: object) -> object:
        return self._queue.pop(0)


class _FakeModel:
    def __init__(self, queue: list[object]) -> None:
        self._queue = queue

    def with_structured_output(self, _schema: object) -> _Structured:
        return _Structured(self._queue)


@pytest.fixture()
def adapter() -> BackendAdapter:
    base_dir = Path(".test_data") / f"analysis_{uuid.uuid4().hex}"
    try:
        yield BackendAdapter(config=BackendConfig(base_data_dir=base_dir / ".data_agent"))
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


_GOOD_CODE = GeneratedAnalysisCode(
    rationale="revenue by category",
    code=(
        "by_cat = df.groupby('category')['revenue'].sum().sort_values(ascending=False)\n"
        "top = str(by_cat.index[0])\n"
        "result = {'summary': f'top category {top}', 'findings': [f'top category {top}'], "
        "'statistics': {'top_category': top}, 'method_decision': {'selected_method': 'grouped sum', 'rationale': 'The objective is to compare category revenue.', 'assumptions_checked': [], 'fallbacks_considered': []}, 'limitations': ['single run']}\n"
    ),
)
_REVIEW_CODE = GeneratedAnalysisCode(
    rationale="revenue by category with actionable review request",
    code=(
        "by_cat = df.groupby('category')['revenue'].sum().sort_values(ascending=False)\n"
        "top = str(by_cat.index[0])\n"
        "result = {\n"
        "  'summary': f'top category {top}',\n"
        "  'findings': [f'top category {top}'],\n"
        "  'statistics': {'top_category': top},\n"
        "  'method_decision': {'selected_method': 'grouped sum', 'rationale': 'The objective is to compare category revenue.', 'assumptions_checked': [], 'fallbacks_considered': []},\n"
        "  'evidence_tables': [{'title': 'category revenue', 'columns': ['category', 'revenue'], 'rows': [{'category': str(k), 'revenue': int(v)} for k, v in by_cat.items()]}],\n"
        "  'review_request': {\n"
        "    'decision_type': 'segment_definition',\n"
        "    'question': 'Use the top revenue category as the follow-up segment?',\n"
        "    'proposal': 'Use the top revenue category as the segment for follow-up analysis.',\n"
        "    'rationale': ['The selected category has the highest observed revenue.'],\n"
        "    'evidence': {'top_category': top},\n"
        "    'options': [\n"
        "      {'id': 'top_only', 'label': 'Use top revenue category', 'method': 'top-category focus', 'assumptions': ['The largest category is the operational focus.'], 'advantages': ['Creates a focused follow-up.'], 'limitations': ['Excludes other categories.'], 'impact': 'Follow-up focuses on one category.', 'recommended': True},\n"
        "      {'id': 'all_categories', 'label': 'Compare all categories', 'method': 'full comparison', 'assumptions': ['All categories remain relevant.'], 'advantages': ['Retains context.'], 'limitations': ['Less focused follow-up.'], 'impact': 'Follow-up compares every category.', 'recommended': False}\n"
        "    ],\n"
        "    'recommended_option_id': 'top_only',\n"
        "    'impact_if_approved': 'Follow-up analysis will focus on the selected segment.',\n"
        "    'requires_followup_analysis': True\n"
        "  },\n"
        "  'limitations': ['single run']\n"
        "}\n"
    ),
)


def test_agent_registers_structured_artifact_and_lineage(adapter: BackendAdapter) -> None:
    run = adapter.create_run()
    sql_ref = adapter.register_artifact(
        run.run_id,
        ArtifactType.sql_result,
        content_text="category,revenue\nA,10\nA,12\nB,20\nB,22\n",
        filename="result.csv",
        created_by_tool="test.sql",
        preview={"row_count": 4, "columns": ["category", "revenue"]},
    )
    state = OrchestrationState(
        run_id=run.run_id,
        user_query="which category has the most revenue",
        goal="which category has the most revenue",
        route_kind="comprehensive",
        plan=AnalysisPlan(goal="revenue by category", metric="revenue", dimension="category", route_kind="comprehensive"),
    )
    state.artifact_ids = {"sql_agent": [sql_ref.artifact_id]}

    envelope = AnalysisAgent().run(
        state,
        AgentRuntime(adapter),
        planner_model=_FakeModel([AnalysisIntent(objective="revenue by category", domain="finance", metric_hints=["revenue"], dimension_hints=["category"])]),
        code_generator_model=_FakeModel([_GOOD_CODE]),
        critic_model=_FakeModel([CodeCritique(verdict="pass")]),
    )

    artifact = adapter.get_artifact(envelope.artifact_ids()[0])
    payload = json.loads(adapter.read_artifact_text(artifact.artifact_id))

    parsed = AnalysisResult.model_validate(payload)
    assert artifact.parent_ids == [sql_ref.artifact_id]
    assert parsed.evidence[0].statistics["top_category"] == "B"
    assert parsed.intent.domain == "finance"
    assert parsed.answer_coverage.coverage_status == "full"
    assert parsed.answer_coverage.used_metrics == ["revenue"]
    assert parsed.answer_coverage.used_dimensions == ["category"]
    assert "top = str(by_cat.index[0])" in parsed.generated_code
    assert parsed.code_critique is None
    assert parsed.debug_artifact_id is not None
    debug_payload = json.loads(adapter.read_artifact_text(parsed.debug_artifact_id))
    assert "top = str(by_cat.index[0])" in debug_payload["generated_code"]
    assert debug_payload["code_critique"]["verdict"] == "pass"
    generated_code_artifacts = [
        item
        for item in adapter.list_artifacts(run_id=run.run_id, artifact_type=ArtifactType.file)
        if item.metadata.get("kind") == "analysis_generated_code"
    ]
    assert len(generated_code_artifacts) == 1
    generated_code_artifact = generated_code_artifacts[0]
    assert generated_code_artifact.parent_ids == [sql_ref.artifact_id]
    assert str(generated_code_artifact.local_path).endswith("analysis_generated_code_attempt_1.py")
    assert "top = str(by_cat.index[0])" in adapter.read_artifact_text(generated_code_artifact.artifact_id)
    progress_events = adapter.services.run_service.list_events(run.run_id)
    generate_completed = next(
        event
        for event in progress_events
        if event.event_type == "analysis.progress"
        and event.metadata.get("stage") == "generate"
        and event.metadata.get("status") == "completed"
    )
    assert generate_completed.metadata["generated_code_artifact_id"] == generated_code_artifact.artifact_id
    assert all(check.passed for check in envelope.validation.local_checks)


def test_agent_skips_binary_eda_chart_artifacts(adapter: BackendAdapter) -> None:
    run = adapter.create_run()
    sql_ref = adapter.register_artifact(
        run.run_id,
        ArtifactType.sql_result,
        content_text="category,revenue\nA,10\nA,12\nB,20\nB,22\n",
        filename="result.csv",
        created_by_tool="test.sql",
        preview={"row_count": 4, "columns": ["category", "revenue"]},
    )
    eda_ref = adapter.register_artifact(
        run.run_id,
        ArtifactType.data_profile,
        content_text=json.dumps({"profile": {"quality_status": "usable"}, "key_charts": []}),
        filename="eda_summary.json",
        created_by_tool="test.eda",
        parent_ids=[sql_ref.artifact_id],
        metadata={"kind": "eda_summary"},
    )
    chart_ref = adapter.register_artifact(
        run.run_id,
        ArtifactType.chart,
        content_bytes=b"\x89PNG\r\n\x1a\nfake-png",
        filename="chart.png",
        created_by_tool="test.eda",
        parent_ids=[sql_ref.artifact_id],
    )
    state = OrchestrationState(
        run_id=run.run_id,
        user_query="which category has the most revenue",
        goal="which category has the most revenue",
        route_kind="comprehensive",
        plan=AnalysisPlan(goal="revenue by category", metric="revenue", dimension="category", route_kind="comprehensive"),
    )
    state.artifact_ids = {"sql_agent": [sql_ref.artifact_id], "eda_agent": [eda_ref.artifact_id, chart_ref.artifact_id]}

    envelope = AnalysisAgent().run(
        state,
        AgentRuntime(adapter),
        planner_model=_FakeModel([AnalysisIntent(objective="revenue by category", domain="finance", metric_hints=["revenue"], dimension_hints=["category"])]),
        code_generator_model=_FakeModel([_GOOD_CODE]),
        critic_model=_FakeModel([CodeCritique(verdict="pass")]),
    )

    public_payload = json.loads(adapter.read_artifact_text(envelope.artifact_ids()[0]))
    parsed = AnalysisResult.model_validate(public_payload)
    assert envelope.status == AgentStatus.success
    assert parsed.evidence[0].statistics["top_category"] == "B"


def test_agent_reads_chart_when_numeric_summary_loses_shape_information(adapter: BackendAdapter) -> None:
    run = adapter.create_run()
    sql_ref = adapter.register_artifact(
        run.run_id,
        ArtifactType.sql_result,
        content_text="category,revenue\nA,10\nA,12\nB,20\nB,220\n",
        filename="result.csv",
        created_by_tool="test.sql",
        preview={"row_count": 4, "columns": ["category", "revenue"]},
    )
    chart_ref = adapter.register_artifact(
        run.run_id,
        ArtifactType.chart,
        content_bytes=b"\x89PNG\r\n\x1a\nfake-png",
        filename="revenue_dist.png",
        created_by_tool="test.eda",
        parent_ids=[sql_ref.artifact_id],
    )
    eda_ref = adapter.register_artifact(
        run.run_id,
        ArtifactType.data_profile,
        content_text=json.dumps({
            "profile": {
                "quality_status": "usable",
                "numeric_summary": {
                    "revenue": {
                        "count": 4.0,
                        "mean": 65.5,
                        "std": 103.1,
                        "min": 10.0,
                        "25%": 11.5,
                        "50%": 16.0,
                        "75%": 70.0,
                        "max": 220.0,
                    }
                },
            },
            "key_charts": [{
                "filename": "revenue_dist.png",
                "artifact_id": chart_ref.artifact_id,
                "chart_type": "histogram",
            }],
        }),
        filename="eda_summary.json",
        created_by_tool="test.eda",
        parent_ids=[sql_ref.artifact_id, chart_ref.artifact_id],
        metadata={"kind": "eda_summary"},
    )
    state = OrchestrationState(
        run_id=run.run_id,
        user_query="which category has the most revenue",
        goal="which category has the most revenue",
        route_kind="comprehensive",
        plan=AnalysisPlan(goal="revenue by category", metric="revenue", dimension="category", route_kind="comprehensive"),
    )
    state.artifact_ids = {"sql_agent": [sql_ref.artifact_id], "eda_agent": [eda_ref.artifact_id, chart_ref.artifact_id]}
    loaded_artifacts: list[str] = []

    def loader(artifact_id: str) -> bytes:
        loaded_artifacts.append(artifact_id)
        return adapter.read_artifact_bytes(artifact_id)

    def reader(chart_images: list[dict], _state: dict) -> list[dict]:
        return [
            {
                "status": "read_success",
                "multimodal_summary": f"{item['chart']['filename']} read with {len(item['image_bytes'])} bytes",
                "cautions": [],
            }
            for item in chart_images
        ]

    envelope = AnalysisAgent().run(
        state,
        AgentRuntime(adapter),
        planner_model=_FakeModel([AnalysisIntent(objective="revenue by category", domain="finance", metric_hints=["revenue"], dimension_hints=["category"])]),
        code_generator_model=_FakeModel([_GOOD_CODE]),
        critic_model=_FakeModel([CodeCritique(verdict="pass")]),
        chart_artifact_loader=loader,
        chart_reader=reader,
    )

    public_payload = json.loads(adapter.read_artifact_text(envelope.artifact_ids()[0]))
    parsed = AnalysisResult.model_validate(public_payload)
    assert loaded_artifacts == [chart_ref.artifact_id]
    assert parsed.chart_status == "read_success"
    assert parsed.chart_requests[0]["related_keys"] == ["profile.numeric_summary.revenue"]
    assert parsed.chart_requests[0]["information_loss"]
    assert parsed.visual_evidence[0].status == "read_success"


def test_agent_routes_method_review_failure_to_retry_not_approval(adapter: BackendAdapter) -> None:
    run = adapter.create_run()
    sql_ref = adapter.register_artifact(
        run.run_id,
        ArtifactType.sql_result,
        content_text="category,revenue\nA,10\nA,12\nB,20\nB,22\n",
        filename="result.csv",
        created_by_tool="test.sql",
        preview={"row_count": 4, "columns": ["category", "revenue"]},
    )
    state = OrchestrationState(
        run_id=run.run_id,
        user_query="which category has the most revenue",
        goal="which category has the most revenue",
        route_kind="comprehensive",
        plan=AnalysisPlan(goal="revenue by category", metric="revenue", dimension="category", route_kind="comprehensive"),
        max_retry_per_agent=1,
    )
    state.artifact_ids = {"sql_agent": [sql_ref.artifact_id]}

    envelope = AnalysisAgent().run(
        state,
        AgentRuntime(adapter),
        planner_model=_FakeModel([AnalysisIntent(objective="revenue by category", domain="finance", metric_hints=["revenue"], dimension_hints=["category"])]),
        code_generator_model=_FakeModel([_GOOD_CODE, _GOOD_CODE, _GOOD_CODE]),
        critic_model=_FakeModel([CodeCritique(verdict="fail", feedback="wrong method")] * 3),
    )

    artifact = adapter.get_artifact(envelope.artifact_ids()[0])
    payload = json.loads(adapter.read_artifact_text(artifact.artifact_id))
    parsed = AnalysisResult.model_validate(payload)

    assert parsed.human_review.required is False
    assert parsed.status == "success"
    assert envelope.status == AgentStatus.success
    assert envelope.approval.required is False
    assert envelope.retry_hint.retryable is False
    assert envelope.retry_hint.reason_code == "none"
    assert envelope.error == ""
    assert any("wrong method" in note for note in parsed.method_notes)
    assert any(
        finding.code == "analysis_method_note"
        and finding.disposition == "limitation"
        and finding.retryable is False
        and "wrong method" in finding.message
        for finding in envelope.validation.findings
    )
    assert envelope.retry_hint.details == {}


def test_agent_review_required_registers_public_and_debug_artifacts(adapter: BackendAdapter) -> None:
    run = adapter.create_run()
    sql_ref = adapter.register_artifact(
        run.run_id,
        ArtifactType.sql_result,
        content_text="category,revenue\nA,10\nA,12\nB,20\nB,22\n",
        filename="result.csv",
        created_by_tool="test.sql",
        preview={"row_count": 4, "columns": ["category", "revenue"]},
    )
    state = OrchestrationState(
        run_id=run.run_id,
        user_query="which category has the most revenue",
        goal="which category has the most revenue",
        route_kind="comprehensive",
        plan=AnalysisPlan(goal="revenue by category", metric="revenue", dimension="category", route_kind="comprehensive"),
    )
    state.artifact_ids = {"sql_agent": [sql_ref.artifact_id]}

    envelope = AnalysisAgent().run(
        state,
        AgentRuntime(adapter),
        planner_model=_FakeModel([AnalysisIntent(objective="revenue by category", domain="finance", metric_hints=["revenue"], dimension_hints=["category"])]),
        code_generator_model=_FakeModel([_REVIEW_CODE]),
        critic_model=_FakeModel([CodeCritique(verdict="review_required", feedback="review operational threshold")]),
    )

    assert envelope.status == AgentStatus.success
    assert envelope.approval.required is True
    public_id, debug_id = envelope.artifact_ids()
    public_payload = json.loads(adapter.read_artifact_text(public_id))
    debug_payload = json.loads(adapter.read_artifact_text(debug_id))
    parsed = AnalysisResult.model_validate(public_payload)

    assert parsed.status == "review_required"
    assert parsed.review_request is not None
    assert parsed.human_review.reason == "Use the top revenue category as the follow-up segment?"
    assert parsed.debug_artifact_id == debug_id
    assert "review_request" in parsed.generated_code
    assert debug_payload["generated_code"]
    assert debug_payload["code_critique"]["verdict"] == "review_required"
    artifact = adapter.get_artifact(public_id)
    assert artifact.preview["review_request"]["recommended_option_id"] == "top_only"
    assert not parsed.key_findings[0].startswith("SQL result contains")
    assert any(
        finding.code == "analysis_review_request"
        and finding.disposition == "semantic_evidence"
        for finding in envelope.validation.findings
    )


def test_agent_non_actionable_review_note_stays_success(adapter: BackendAdapter) -> None:
    run = adapter.create_run()
    sql_ref = adapter.register_artifact(
        run.run_id,
        ArtifactType.sql_result,
        content_text="category,revenue\nA,10\nA,12\nB,20\nB,22\n",
        filename="result.csv",
        created_by_tool="test.sql",
        preview={"row_count": 4, "columns": ["category", "revenue"]},
    )
    state = OrchestrationState(
        run_id=run.run_id,
        user_query="which category has the most revenue",
        goal="which category has the most revenue",
        route_kind="comprehensive",
        plan=AnalysisPlan(goal="revenue by category", metric="revenue", dimension="category", route_kind="comprehensive"),
    )
    state.artifact_ids = {"sql_agent": [sql_ref.artifact_id]}

    envelope = AnalysisAgent().run(
        state,
        AgentRuntime(adapter),
        planner_model=_FakeModel([AnalysisIntent(objective="revenue by category", domain="finance", metric_hints=["revenue"], dimension_hints=["category"])]),
        code_generator_model=_FakeModel([_GOOD_CODE]),
        critic_model=_FakeModel([CodeCritique(verdict="review_required", feedback="sample size is small")]),
    )

    public_id, debug_id = envelope.artifact_ids()
    public_payload = json.loads(adapter.read_artifact_text(public_id))
    debug_payload = json.loads(adapter.read_artifact_text(debug_id))
    parsed = AnalysisResult.model_validate(public_payload)

    assert envelope.status == AgentStatus.success
    assert envelope.approval.required is False
    assert parsed.status == "success"
    assert parsed.review_request is None
    assert parsed.method_notes == ["sample size is small"]
    assert debug_payload["method_notes"] == ["sample size is small"]
    assert any(
        finding.code == "analysis_method_note"
        and finding.disposition == "limitation"
        and finding.message == "sample size is small"
        for finding in envelope.validation.findings
    )

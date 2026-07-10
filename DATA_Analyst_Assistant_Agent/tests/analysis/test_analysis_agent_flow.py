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
        "'statistics': {'top_category': top}, 'limitations': ['single run']}\n"
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
    assert all(check.passed for check in envelope.validation.local_checks)

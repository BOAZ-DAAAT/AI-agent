from __future__ import annotations

import shutil
import uuid
import json
from pathlib import Path

import pytest

from data_agent_backend.config import BackendConfig
from data_agent_backend.models.artifacts import ArtifactType
from DATA_Analyst_Assistant_Agent import BackendAdapter, SQLAgentSupervisor, SupervisorTerminalState
from DATA_Analyst_Assistant_Agent.agents.analysis import AnalysisAgent
from DATA_Analyst_Assistant_Agent.agents.eda import EDAAgent
from DATA_Analyst_Assistant_Agent.agents.report import ReportAgent
from DATA_Analyst_Assistant_Agent.agents.validation import CentralValidationAgent
from DATA_Analyst_Assistant_Agent.agents.visualization.chart_selector import build_chart_config
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.shared.contracts import AgentEnvelope, AnalysisPlan, OrchestrationState


@pytest.fixture()
def adapter() -> BackendAdapter:
    base_dir = Path(".test_data") / f"phase2_c_{uuid.uuid4().hex}"
    config = BackendConfig(base_data_dir=base_dir / ".data_agent")
    try:
        yield BackendAdapter(config=config)
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def artifact_text(adapter: BackendAdapter, artifact_id: str) -> str:
    artifact = adapter.get_artifact(artifact_id)
    assert artifact.local_path is not None
    return Path(artifact.local_path).read_text(encoding="utf-8")


def register_sql_csv(adapter: BackendAdapter, run_id: str, csv_text: str, *, row_count: int = 2) -> str:
    ref = adapter.register_artifact(
        run_id,
        ArtifactType.sql_result,
        content_text=csv_text,
        filename="result.csv",
        created_by_tool="test.sql",
        preview={"row_count": row_count, "columns": csv_text.splitlines()[0].split(","), "sample_rows": []},
    )
    return ref.artifact_id


def test_eda_profile_preview_includes_phase2_quality_fields(adapter: BackendAdapter) -> None:
    supervisor = SQLAgentSupervisor(adapter)
    state = supervisor.run("Profile data quality and missing values, then make a report.")

    eda_artifact = adapter.get_artifact(state.artifact_ids["eda_agent"][0])

    assert eda_artifact.type == ArtifactType.data_profile
    assert eda_artifact.preview["row_count"] >= 1
    assert len(eda_artifact.preview["columns"]) >= 1
    assert eda_artifact.preview["sample_available"] is True
    assert eda_artifact.preview["quality_status"] == "usable"
    assert eda_artifact.preview["key_issues"] == []
    assert eda_artifact.preview["recommended_next_steps"] == ["continue_to_analysis", "document_limitations"]


def test_eda_reads_sql_result_csv_body_into_profile(adapter: BackendAdapter) -> None:
    run = adapter.create_run()
    sql_id = register_sql_csv(adapter, run.run_id, "category,revenue\nA,10\nB,20\nA,\n", row_count=3)
    state = OrchestrationState(
        run_id=run.run_id,
        user_query="profile csv",
        artifact_ids={"sql_agent": [sql_id]},
        plan=AnalysisPlan(goal="profile csv", route_kind="eda"),
        route_kind="eda",
    )

    envelope = EDAAgent().run(state, AgentRuntime(adapter))
    payload = json.loads(artifact_text(adapter, envelope.artifact_ids()[0]))
    profile = payload["profile"]

    assert profile["row_count"] == 3
    assert profile["dtypes"]["category"] == "object"
    assert profile["null_counts"]["revenue"] == 1
    assert profile["unique_counts"]["category"] == 2
    assert profile["numeric_summary"]["revenue"]["mean"] == 15.0
    assert profile["categorical_top_values"]["category"]["A"] == 2


def test_analysis_uses_eda_profile_when_available(adapter: BackendAdapter) -> None:
    supervisor = SQLAgentSupervisor(adapter)
    state = supervisor.run("Profile data quality and missing values, then make a report.")

    envelope = AnalysisAgent().run(state, AgentRuntime(adapter))
    analysis_artifact = adapter.get_artifact(envelope.artifact_ids()[0])

    assert analysis_artifact.metadata["kind"] == "analysis_result"
    assert "EDA profile evidence was considered" in " ".join(analysis_artifact.preview["key_findings"])
    assert "EDA profile artifacts summarize data quality signals" in " ".join(analysis_artifact.preview["limitations"])


def test_analysis_uses_sql_csv_and_eda_profile_artifacts(adapter: BackendAdapter) -> None:
    run = adapter.create_run()
    sql_id = register_sql_csv(adapter, run.run_id, "category,revenue\nA,10\nB,20\n", row_count=2)
    state = OrchestrationState(
        run_id=run.run_id,
        user_query="analyze revenue by category",
        artifact_ids={"sql_agent": [sql_id]},
        plan=AnalysisPlan(goal="analyze", metric="revenue", dimension="category", route_kind="trend"),
        route_kind="trend",
    )
    eda_envelope = EDAAgent().run(state, AgentRuntime(adapter))
    state.add_artifacts("eda_agent", eda_envelope.artifact_ids())

    analysis_envelope = AnalysisAgent().run(state, AgentRuntime(adapter))
    result = json.loads(artifact_text(adapter, analysis_envelope.artifact_ids()[0]))

    assert "SQL result contains 2 rows and 2 columns." in result["key_findings"]
    assert "Top category by revenue is B (20)." in result["key_findings"]
    assert result["source_artifacts"]["sql"] == [sql_id]
    assert result["source_artifacts"]["eda"] == eda_envelope.artifact_ids()


def test_trend_chart_config_includes_axis_and_references(adapter: BackendAdapter) -> None:
    supervisor = SQLAgentSupervisor(adapter)
    state = supervisor.run("Analyze monthly revenue trend with a chart.")

    chart_artifact = adapter.get_artifact(state.artifact_ids["visualization_agent"][0])
    chart_payload = artifact_text(adapter, chart_artifact.artifact_id)

    assert chart_artifact.preview["chart_type"] == "line"
    assert chart_artifact.preview["title"]
    assert chart_artifact.preview["encoding"]["x"]["unit"] == "unknown"
    assert chart_artifact.preview["encoding"]["y"]["unit"] == "unknown"
    assert chart_artifact.preview["data_reference"] == state.artifact_ids["sql_agent"]
    assert "unit_note" in chart_payload


def test_visualization_selects_route_chart_types_from_state_and_data() -> None:
    import pandas as pd

    trend_state = OrchestrationState(run_id="run_chart", user_query="monthly revenue", route_kind="trend")
    bar_state = OrchestrationState(run_id="run_chart", user_query="top category", route_kind="simple")
    histogram_state = OrchestrationState(run_id="run_chart", user_query="distribution of revenue", route_kind="simple")
    table_state = OrchestrationState(run_id="run_chart", user_query="single KPI", route_kind="simple")

    df = pd.DataFrame({"category": ["A", "B"], "revenue": [10, 20]})
    numeric_df = pd.DataFrame({"revenue": [10, 20, 30]})
    single_df = pd.DataFrame({"revenue": [10]})

    assert build_chart_config(trend_state, dataframe=df)["chart_type"] == "line"
    assert build_chart_config(bar_state, dataframe=df)["chart_type"] == "bar"
    assert build_chart_config(histogram_state, dataframe=numeric_df)["chart_type"] == "histogram"
    assert build_chart_config(table_state, dataframe=single_df)["chart_type"] == "table"


def test_report_sections_and_evidence_include_upstream_artifacts(adapter: BackendAdapter) -> None:
    supervisor = SQLAgentSupervisor(adapter)
    state = supervisor.run("Analyze monthly revenue trend with a chart.")
    report = artifact_text(adapter, state.artifact_ids["report_agent"][0])

    for heading in ("## Summary", "## Key Findings", "## Evidence", "## Visuals", "## Limitations", "## Next Actions"):
        assert heading in report

    for agent_name in ("sql_agent", "analysis_agent", "visualization_agent"):
        for artifact_id in state.artifact_ids[agent_name]:
            assert f"- {agent_name}: {artifact_id}" in report


def test_report_includes_question_sql_eda_analysis_viz_and_evidence(adapter: BackendAdapter) -> None:
    supervisor = SQLAgentSupervisor(adapter)
    state = supervisor.run("Profile data quality and analyze monthly revenue trend with a chart.")
    report = artifact_text(adapter, state.artifact_ids["report_agent"][0])

    for heading in ("## User Question", "## Generated SQL", "## EDA Summary", "## Key Findings", "## Visuals", "## Evidence"):
        assert heading in report
    assert state.user_query in report
    assert state.generated_sql in report
    for agent_name in ("sql_agent", "eda_agent", "analysis_agent", "visualization_agent"):
        for artifact_id in state.artifact_ids[agent_name]:
            assert artifact_id in report


def test_validation_flags_row_count_quality_and_missing_report_evidence(adapter: BackendAdapter) -> None:
    run = adapter.create_run()
    sql_id = register_sql_csv(adapter, run.run_id, "category,revenue\n", row_count=0)
    eda_ref = adapter.register_artifact(
        run.run_id,
        ArtifactType.data_profile,
        content_text=json.dumps({"profile": {"quality_status": "needs_review", "key_issues": ["empty result"]}}),
        filename="eda_summary.json",
        created_by_tool="test.eda",
    )
    report_ref = adapter.save_workspace_file(run.run_id, "# Report\n\n## Evidence\n- nothing useful")
    state = OrchestrationState(
        run_id=run.run_id,
        user_query="validate",
        artifact_ids={"sql_agent": [sql_id], "eda_agent": [eda_ref.artifact_id], "report_agent": [report_ref.artifact_id]},
    )
    upstream = AgentEnvelope(agent_name="report_agent", summary="report", artifact_refs=[report_ref])

    envelope = CentralValidationAgent().run(state, AgentRuntime(adapter), upstream)
    payload = json.loads(artifact_text(adapter, envelope.artifact_ids()[0]))
    categories = {finding["category"] for finding in payload["findings"]}

    assert {"empty_sql_result", "eda_quality_status", "report_evidence_missing"} <= categories
    assert payload["verdict"] == "fail"
    assert payload["retryable"] is True


def test_mart_approval_waiting_flow_does_not_create_report(adapter: BackendAdapter) -> None:
    supervisor = SQLAgentSupervisor(adapter)
    state = supervisor.run("반복 조회용 데이터마트 저장을 제안해줘.")

    assert state.terminal_state == SupervisorTerminalState.needs_user_approval
    assert "report_agent" not in state.artifact_ids

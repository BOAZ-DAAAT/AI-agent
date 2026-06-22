from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest

from data_agent_backend.config import BackendConfig
from data_agent_backend.models.artifacts import ArtifactType
from DATA_Analyst_Assistant_Agent.agents.analysis import AnalysisAgent, AnalysisExecutionPlan, AnalysisResult
from DATA_Analyst_Assistant_Agent.agents.analysis.context import build_analysis_context
from DATA_Analyst_Assistant_Agent.agents.analysis.methods import build_analysis_result
from DATA_Analyst_Assistant_Agent.agents.analysis.planner import build_analysis_plan
from DATA_Analyst_Assistant_Agent.agents.analysis.tools import ANALYSIS_TOOLS
from DATA_Analyst_Assistant_Agent.agents.analysis.workflow import run_analysis_workflow
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.shared.contracts import AnalysisPlan, OrchestrationState


class FakeStructuredModel:
    def __init__(self, response: AnalysisExecutionPlan, error: Exception | None = None) -> None:
        self.response = response
        self.error = error

    def invoke(self, messages):
        if self.error:
            raise self.error
        assert len(messages) == 2
        return self.response


class FakeChatModel:
    def __init__(self, response: AnalysisExecutionPlan, error: Exception | None = None) -> None:
        self.response = response
        self.error = error

    def with_structured_output(self, schema):
        assert schema is AnalysisExecutionPlan
        return FakeStructuredModel(self.response, self.error)


def _execution_plan(
    *,
    question_type: str = "comparison",
    analysis_kind: str = "group_comparison",
    analysis_subtype: str = "welch_t_test_or_one_way_anova",
    tool_names: list[str] | None = None,
    metric: str | None = "revenue",
    dimension: str | None = "category",
    feature_columns: list[str] | None = None,
    requires_human_review: bool = False,
    review_reason: str = "",
    time_column: str | None = None,
    tool_parameters: dict | None = None,
) -> AnalysisExecutionPlan:
    return AnalysisExecutionPlan(
        objective="analyze",
        question_type=question_type,
        analysis_kind=analysis_kind,
        analysis_subtype=analysis_subtype,
        tool_names=tool_names or ["describe_metric", "compare_groups", "test_group_difference"],
        metric=metric,
        dimension=dimension,
        time_column=time_column,
        feature_columns=feature_columns or ([dimension] if dimension else []),
        tool_parameters=tool_parameters or {},
        requires_human_review=requires_human_review,
        review_reason=review_reason,
        planner_mode="llm",
    )


@pytest.fixture()
def adapter() -> BackendAdapter:
    base_dir = Path(".test_data") / f"analysis_{uuid.uuid4().hex}"
    try:
        yield BackendAdapter(config=BackendConfig(base_data_dir=base_dir / ".data_agent"))
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def _state(run_id: str, query: str = "analyze revenue by category") -> OrchestrationState:
    return OrchestrationState(
        run_id=run_id,
        user_query=query,
        goal=query,
        route_kind="comprehensive",
        plan=AnalysisPlan(goal=query, metric="revenue", dimension="category", route_kind="comprehensive"),
    )


def test_tools_are_langchain_tools_with_stable_names() -> None:
    assert set(ANALYSIS_TOOLS) == {
        "describe_metric",
        "compare_groups",
        "measure_correlation",
        "test_group_difference",
        "analyze_trend",
        "fit_regression_model",
        "fit_classification_model",
        "detect_anomalies",
        "analyze_time_series",
        "analyze_cohort_retention",
        "analyze_funnel",
        "analyze_journey",
        "segment_entities",
        "analyze_rfm",
        "analyze_contribution",
        "analyze_mix_shift",
        "analyze_survival",
        "analyze_text",
        "simulate_scenario",
        "run_bayesian_mmm",
        "estimate_probabilistic_clv",
        "analyze_geospatial_hotspots",
        "optimize_business_allocation",
    }
    assert all(tool.args_schema is not None for tool in ANALYSIS_TOOLS.values())


def test_context_is_bounded_and_plan_uses_real_columns() -> None:
    state = _state("run_context")
    df = pd.DataFrame({"category": [f"C{i}" for i in range(20)], "revenue": range(20)})
    context = build_analysis_context(state, df, [], sample_limit=3)
    plan = build_analysis_plan(context, model=FakeChatModel(_execution_plan()))

    assert len(context.sample_rows) == 3
    assert context.column_profiles["category"]["unique_count"] == 20
    assert context.column_profiles["revenue"]["mean"] == 9.5
    assert plan.metric == "revenue"
    assert plan.dimension == "category"
    assert plan.tool_names == ["describe_metric", "compare_groups", "test_group_difference"]


def test_structured_result_contains_traceable_evidence() -> None:
    state = _state("run_result")
    df = pd.DataFrame({"category": ["A", "B", "A"], "revenue": [10, 20, 5]})
    payload = build_analysis_result(state, dataframe=df, execution_plan=_execution_plan())
    result = AnalysisResult.model_validate(payload)

    assert result.plan.analysis_kind.value == "group_comparison"
    assert {item.tool_name for item in result.evidence} == {"describe_metric", "compare_groups"}
    assert "SQL result contains 3 rows and 2 columns." in result.key_findings
    assert "Top category by revenue is B (20)." in result.key_findings
    assert result.hypotheses[0].decision == "inconclusive"


def test_causal_request_emits_human_review_requirement() -> None:
    state = _state("run_review", "What is the causal impact of category on revenue?")
    df = pd.DataFrame({"category": ["A", "B"], "revenue": [10, 20]})
    payload = build_analysis_result(
        state,
        dataframe=df,
        question_type="causal",
        execution_plan=_execution_plan(
            question_type="causal",
            analysis_kind="correlation",
            analysis_subtype="pearson_correlation",
            tool_names=["measure_correlation"],
            dimension=None,
            feature_columns=["revenue"],
            requires_human_review=True,
            review_reason="Causal interpretation requires human review.",
        ),
    )

    assert payload["human_review"]["required"] is True
    assert "causal" in payload["human_review"]["reason"].casefold()


def test_explicit_prediction_type_selects_regression_subtype() -> None:
    state = _state("run_prediction", "analyze this dataset")
    df = pd.DataFrame({
        "category": ["A", "B"] * 10,
        "revenue": [float(index * 3 + (index % 2)) for index in range(20)],
        "orders": list(range(20)),
    })
    payload = build_analysis_result(
        state,
        dataframe=df,
        question_type="prediction",
        execution_plan=_execution_plan(
            question_type="prediction",
            analysis_kind="regression",
            analysis_subtype="linear_regression_baseline",
            tool_names=["describe_metric", "fit_regression_model"],
            dimension=None,
            feature_columns=["category", "orders"],
        ),
    )

    assert payload["plan"]["question_type"] == "prediction"
    assert payload["plan"]["analysis_kind"] == "regression"
    assert payload["plan"]["analysis_subtype"] == "linear_regression_baseline"
    assert "fit_regression_model" in payload["plan"]["tool_names"]
    regression = next(item for item in payload["evidence"] if item["tool_name"] == "fit_regression_model")
    assert regression["statistics"]["evaluation_scope"] == "fixed_holdout"
    assert {"test_r2", "test_mae", "test_rmse", "top_coefficients"} <= set(regression["statistics"])


def test_explicit_anomaly_type_overrides_generic_question() -> None:
    state = _state("run_anomaly", "analyze this dataset")
    df = pd.DataFrame({
        "revenue": [10.0] * 19 + [1000.0],
        "orders": list(range(20)),
    })
    payload = build_analysis_result(
        state,
        dataframe=df,
        question_type="anomaly_detection",
        execution_plan=_execution_plan(
            question_type="anomaly_detection",
            analysis_kind="anomaly_detection",
            analysis_subtype="isolation_forest",
            tool_names=["detect_anomalies"],
            metric=None,
            dimension=None,
            feature_columns=["revenue", "orders"],
        ),
    )

    assert payload["plan"]["question_type"] == "anomaly_detection"
    assert payload["plan"]["analysis_kind"] == "anomaly_detection"
    assert payload["plan"]["analysis_subtype"] == "isolation_forest"
    anomaly = next(item for item in payload["evidence"] if item["tool_name"] == "detect_anomalies")
    assert anomaly["statistics"]["anomaly_count"] >= 1
    assert anomaly["statistics"]["top_anomaly_candidates"]


def test_explicit_classification_type_selects_logistic_subtype() -> None:
    state = _state("run_classification", "analyze this dataset")
    df = pd.DataFrame({
        "category": ["low", "high"] * 10,
        "revenue": [float(index) for index in range(20)],
        "orders": [index % 4 for index in range(20)],
    })
    payload = build_analysis_result(
        state,
        dataframe=df,
        question_type="classification",
        execution_plan=_execution_plan(
            question_type="classification",
            analysis_kind="classification",
            analysis_subtype="logistic_classification",
            tool_names=["fit_classification_model"],
            metric="category",
            dimension=None,
            feature_columns=["revenue", "orders"],
        ),
    )

    assert payload["plan"]["analysis_kind"] == "classification"
    assert payload["plan"]["analysis_subtype"] == "logistic_classification"
    classification = next(item for item in payload["evidence"] if item["tool_name"] == "fit_classification_model")
    assert classification["statistics"]["evaluation_scope"] == "stratified_fixed_holdout"
    assert {"test_accuracy", "test_balanced_accuracy", "test_f1_weighted"} <= set(classification["statistics"])


def test_time_series_sector_returns_growth_and_trend() -> None:
    state = _state("run_time_series", "analyze monthly revenue")
    dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    df = pd.DataFrame({"sales_month": dates.astype(str), "revenue": [100 + index * 10 for index in range(12)]})
    plan = _execution_plan(
        question_type="time_series",
        analysis_kind="time_series",
        analysis_subtype="monthly_growth_trend",
        tool_names=["analyze_time_series"],
        dimension=None,
        time_column="sales_month",
        tool_parameters={"analyze_time_series": {"frequency": "M", "aggregation": "sum"}},
    )
    payload = build_analysis_result(
        state, dataframe=df, question_type="time_series", execution_plan=plan
    )
    evidence = payload["evidence"][0]["statistics"]

    assert evidence["period_count"] == 12
    assert evidence["total_growth_rate"] > 1.0
    assert evidence["trend_p_value"] < 0.05


def test_cohort_sector_builds_retention_matrix() -> None:
    state = _state("run_cohort", "monthly cohort retention")
    df = pd.DataFrame({
        "customer_id": ["A", "A", "A", "B", "B", "C", "C"],
        "event_time": [
            "2024-01-05", "2024-02-05", "2024-03-05",
            "2024-01-10", "2024-02-10",
            "2024-02-01", "2024-03-01",
        ],
    })
    plan = _execution_plan(
        question_type="cohort",
        analysis_kind="cohort",
        analysis_subtype="monthly_retention",
        tool_names=["analyze_cohort_retention"],
        metric=None,
        dimension=None,
        tool_parameters={"analyze_cohort_retention": {
            "entity_id_column": "customer_id",
            "event_time_column": "event_time",
            "period_unit": "month",
        }},
    )
    payload = build_analysis_result(state, dataframe=df, question_type="cohort", execution_plan=plan)
    evidence = payload["evidence"][0]["statistics"]

    assert evidence["cohort_count"] == 2
    assert evidence["entity_count"] == 3
    assert any(item["period_index"] == 1 for item in evidence["retention_matrix"])


def test_funnel_sector_enforces_ordered_steps() -> None:
    state = _state("run_funnel", "signup purchase funnel")
    df = pd.DataFrame({
        "user_id": ["A", "A", "A", "B", "B", "C", "C", "C"],
        "event": ["visit", "signup", "purchase", "visit", "signup", "signup", "visit", "purchase"],
        "event_time": pd.date_range("2024-01-01", periods=8, freq="h").astype(str),
    })
    plan = _execution_plan(
        question_type="funnel",
        analysis_kind="funnel",
        analysis_subtype="strict_ordered_funnel",
        tool_names=["analyze_funnel"],
        metric=None,
        dimension=None,
        tool_parameters={"analyze_funnel": {
            "entity_id_column": "user_id",
            "event_column": "event",
            "event_time_column": "event_time",
            "steps": ["visit", "signup", "purchase"],
        }},
    )
    payload = build_analysis_result(state, dataframe=df, question_type="funnel", execution_plan=plan)
    evidence = payload["evidence"][0]["statistics"]

    assert evidence["steps"][0]["entities"] == 3
    assert evidence["steps"][1]["entities"] == 2
    assert evidence["steps"][-1]["entities"] == 1
    assert evidence["overall_conversion_rate"] == pytest.approx(1 / 3)


def test_remaining_enterprise_sector_tools_smoke() -> None:
    records = [
        {
            "customer_id": f"C{index}",
            "event_time": f"2024-01-{index + 1:02d}",
            "amount": float(10 + index),
            "feature_a": float(index),
            "feature_b": float(index % 3),
            "duration": float(index + 1),
            "observed": 1 if index % 3 else 0,
            "comment": "fast delivery good service" if index % 2 else "late delivery bad service",
            "period": "2024-01" if index < 5 else "2024-02",
            "category": "A" if index % 2 else "B",
        }
        for index in range(10)
    ]
    rfm = ANALYSIS_TOOLS["analyze_rfm"].invoke({
        "records": records,
        "customer_id_column": "customer_id",
        "event_time_column": "event_time",
        "amount_column": "amount",
    })
    segments = ANALYSIS_TOOLS["segment_entities"].invoke({
        "records": records,
        "entity_id_column": "customer_id",
        "features": ["feature_a", "feature_b"],
        "n_clusters": 2,
    })
    contribution = ANALYSIS_TOOLS["analyze_contribution"].invoke({
        "records": records, "dimension": "category", "metric": "amount",
    })
    mix_shift = ANALYSIS_TOOLS["analyze_mix_shift"].invoke({
        "records": records, "period_column": "period", "dimension": "category", "metric": "amount",
    })
    survival = ANALYSIS_TOOLS["analyze_survival"].invoke({
        "records": records, "duration_column": "duration", "event_observed_column": "observed",
    })
    text = ANALYSIS_TOOLS["analyze_text"].invoke({"records": records, "text_column": "comment"})
    scenario = ANALYSIS_TOOLS["simulate_scenario"].invoke({
        "records": records, "metric": "amount", "change_percent": 10.0,
    })

    assert rfm["customer_count"] == 10
    assert segments["n_clusters"] == 2
    assert contribution["group_count"] == 2
    assert mix_shift["previous_period"] == "2024-01"
    assert survival["observation_count"] == 10
    assert text["top_terms"]
    assert scenario["projected"] == pytest.approx(scenario["baseline"] * 1.1)


def test_geospatial_and_optimization_capabilities_execute() -> None:
    spatial_records = [
        {
            "latitude": 37.50 + index * 0.001,
            "longitude": 127.00 + index * 0.001,
            "sales": 100.0 if index < 6 else 10.0,
        }
        for index in range(12)
    ]
    spatial = ANALYSIS_TOOLS["analyze_geospatial_hotspots"].invoke({
        "records": spatial_records,
        "latitude_column": "latitude",
        "longitude_column": "longitude",
        "metric": "sales",
        "k_neighbors": 3,
        "permutations": 19,
    })
    optimization = ANALYSIS_TOOLS["optimize_business_allocation"].invoke({
        "records": [
            {"channel": "search", "value": 12.0, "cost": 4.0},
            {"channel": "social", "value": 8.0, "cost": 5.0},
        ],
        "item_column": "channel",
        "value_column": "value",
        "cost_column": "cost",
        "budget": 100.0,
        "maximum_allocations": {"search": 15.0, "social": 10.0},
    })

    assert spatial["point_count"] == 12
    assert "global_moran_i" in spatial
    assert optimization["status"] == "optimal"
    assert optimization["budget_used"] <= 100.0
    assert set(optimization["allocations"]) == {"search", "social"}


def test_bayesian_capabilities_enforce_data_contracts() -> None:
    with pytest.raises(ValueError, match="at least 52"):
        ANALYSIS_TOOLS["run_bayesian_mmm"].invoke({
            "records": [
                {"date": "2024-01-01", "sales": 10.0, "search_spend": 2.0},
                {"date": "2024-01-08", "sales": 12.0, "search_spend": 3.0},
            ],
            "date_column": "date",
            "outcome_column": "sales",
            "channel_columns": ["search_spend"],
            "draws": 5,
            "tune": 5,
            "chains": 1,
        })
    with pytest.raises(ValueError, match="at least twenty"):
        ANALYSIS_TOOLS["estimate_probabilistic_clv"].invoke({
            "records": [
                {"customer": "A", "date": "2024-01-01", "amount": 10.0},
                {"customer": "A", "date": "2024-02-01", "amount": 12.0},
            ],
            "customer_id_column": "customer",
            "datetime_column": "date",
            "monetary_value_column": "amount",
            "draws": 5,
            "tune": 5,
            "chains": 1,
        })


def test_internal_workflow_runs_plan_execute_validate_nodes() -> None:
    state = _state("run_workflow")
    df = pd.DataFrame({"category": ["A", "A", "B", "B"], "revenue": [10, 12, 20, 22]})
    result, checks, terminal_reason = run_analysis_workflow(
        state,
        df,
        [],
        question_type="comparison",
        planner_model=FakeChatModel(_execution_plan()),
    )

    assert result["plan"]["question_type"] == "comparison"
    assert result["hypotheses"][0]["decision"] == "supported"
    group_test = next(item for item in result["evidence"] if item["tool_name"] == "test_group_difference")
    assert group_test["statistics"]["effect_size_name"] == "cohen_d"
    assert terminal_reason == "validated_result"
    assert all(check.passed for check in checks)


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
    state = _state(run.run_id)
    state.artifact_ids = {"sql_agent": [sql_ref.artifact_id]}

    envelope = AnalysisAgent().run(
        state,
        AgentRuntime(adapter),
        planner_model=FakeChatModel(_execution_plan()),
    )
    artifact = adapter.get_artifact(envelope.artifact_ids()[0])
    payload = json.loads(adapter.read_artifact_text(artifact.artifact_id))

    AnalysisResult.model_validate(payload)
    assert artifact.parent_ids == [sql_ref.artifact_id]
    assert artifact.preview["tool_names"] == ["describe_metric", "compare_groups", "test_group_difference"]
    assert all(check.passed for check in envelope.validation.local_checks)


def test_llm_failure_is_not_replaced_by_fallback() -> None:
    state = _state("run_llm_failure")
    df = pd.DataFrame({"category": ["A", "B"], "revenue": [10, 20]})
    context = build_analysis_context(state, df, [], question_type="comparison")

    with pytest.raises(RuntimeError, match="planner unavailable"):
        build_analysis_plan(
            context,
            model=FakeChatModel(_execution_plan(), error=RuntimeError("planner unavailable")),
        )


def test_invalid_llm_plan_is_rejected_without_substitution() -> None:
    state = _state("run_invalid_plan")
    df = pd.DataFrame({"category": ["A", "B"], "revenue": [10, 20]})
    context = build_analysis_context(state, df, [], question_type="comparison")
    invalid = _execution_plan(tool_names=["invented_tool"])

    with pytest.raises(ValueError, match="unknown tools"):
        build_analysis_plan(context, model=FakeChatModel(invalid))

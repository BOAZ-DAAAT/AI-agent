from DATA_Analyst_Assistant_Agent.agents.analysis.tools.anomaly import detect_anomalies
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.bayesian import run_bayesian_mmm
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.business import analyze_contribution, analyze_mix_shift, optimize_business_allocation, simulate_scenario
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.comparison import compare_groups, test_group_difference
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.customer import analyze_cohort_retention, analyze_funnel, analyze_journey, analyze_rfm, estimate_probabilistic_clv, segment_entities
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.descriptive import describe_metric
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.geo import analyze_geospatial_hotspots
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.prediction import fit_classification_model, fit_regression_model
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.relationship import measure_correlation
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.survival import analyze_survival
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.text import analyze_text
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.trend import analyze_time_series, analyze_trend
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.catalog import CAPABILITIES, AnalysisCapability

ANALYSIS_TOOLS = {
    item.name: item
    for item in (
        describe_metric,
        compare_groups,
        measure_correlation,
        test_group_difference,
        analyze_trend,
        fit_regression_model,
        fit_classification_model,
        detect_anomalies,
        analyze_time_series,
        analyze_cohort_retention,
        analyze_funnel,
        analyze_journey,
        segment_entities,
        analyze_rfm,
        analyze_contribution,
        analyze_mix_shift,
        analyze_survival,
        analyze_text,
        simulate_scenario,
        run_bayesian_mmm,
        estimate_probabilistic_clv,
        analyze_geospatial_hotspots,
        optimize_business_allocation,
    )
}

__all__ = ["ANALYSIS_TOOLS", "CAPABILITIES", "AnalysisCapability"]

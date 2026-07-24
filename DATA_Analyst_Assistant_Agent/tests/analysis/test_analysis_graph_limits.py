from __future__ import annotations

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.analysis import graph as analysis_graph
from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.analyze import AnalysisOutcome
from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState


def test_analyze_node_uses_two_internal_attempts_independent_of_supervisor_retry(
    monkeypatch,
) -> None:
    captured: dict[str, int] = {}

    def fake_run_analysis(*_args, **kwargs) -> AnalysisOutcome:
        captured["max_attempts"] = kwargs["max_attempts"]
        return AnalysisOutcome(status="passed", attempts=1, result={})

    monkeypatch.setattr(analysis_graph, "run_analysis", fake_run_analysis)
    state = {
        "intent": object(),
        "analysis_context": object(),
        "dataframe": pd.DataFrame({"value": [1]}),
        "orchestration_state": OrchestrationState(
            run_id="run_001",
            user_query="매출 분석",
            max_retry_per_agent=99,
        ),
    }

    result = analysis_graph.analyze_node(state)

    assert result["error"] == ""
    assert captured["max_attempts"] == 2

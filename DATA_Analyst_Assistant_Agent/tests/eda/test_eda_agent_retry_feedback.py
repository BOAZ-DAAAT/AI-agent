from __future__ import annotations

from typing import Any

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda.agent import EDAAgent
from DATA_Analyst_Assistant_Agent.agents.artifact_data import CsvArtifactData
from DATA_Analyst_Assistant_Agent.shared.contracts import AnalysisPlan, OrchestrationState


class _FakeApp:
    def __init__(self) -> None:
        self.invoked_with: dict[str, Any] | None = None

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.invoked_with = payload
        return {"error_log": []}


def _patch_build_app(monkeypatch, fake_app: _FakeApp) -> None:
    monkeypatch.setattr(
        "DATA_Analyst_Assistant_Agent.agents.eda.graph.build_app",
        lambda: fake_app,
    )
    monkeypatch.setattr(
        "DATA_Analyst_Assistant_Agent.agents.eda.lib.visualize.clear_output_dirs",
        lambda: None,
    )


def _csvs() -> list[CsvArtifactData]:
    df = pd.DataFrame({"a": [1, 2, 3]})
    return [CsvArtifactData(artifact_id="a1", text="a\n1\n2\n3\n", dataframe=df, error=None)]


def test_eda_agent_seeds_validation_feedback_from_agent_feedback(monkeypatch) -> None:
    fake_app = _FakeApp()
    _patch_build_app(monkeypatch, fake_app)
    state = OrchestrationState(
        run_id="run_1",
        user_query="상위 20% vs 일반 고객군 만족도 분포 차이를 검정해줘",
        plan=AnalysisPlan(
            goal="고객 세그먼트별 만족도 분포 비교",
            retry_context={
                "agent_feedback": {
                    "eda_agent": {
                        "reason": "그룹별 분포 비교 없이 컬럼 나열만 반복함",
                        "missing_evidence": ["상위20% vs 일반군 기술통계"],
                        "source": "semantic",
                    }
                }
            },
        ),
    )

    EDAAgent()._run_eda_graph(_csvs(), state)

    assert fake_app.invoked_with is not None
    feedback = fake_app.invoked_with["validation_feedback"]
    assert "그룹별 분포 비교 없이 컬럼 나열만 반복함" in feedback
    assert "상위20% vs 일반군 기술통계" in feedback


def test_eda_agent_validation_feedback_empty_without_retry_context(monkeypatch) -> None:
    fake_app = _FakeApp()
    _patch_build_app(monkeypatch, fake_app)
    state = OrchestrationState(run_id="run_1", user_query="월별 매출 추이를 보여줘")

    EDAAgent()._run_eda_graph(_csvs(), state)

    assert fake_app.invoked_with is not None
    assert fake_app.invoked_with["validation_feedback"] == ""

from __future__ import annotations

from typing import Any

from DATA_Analyst_Assistant_Agent.agents.sql.agent import SQLAgent
from DATA_Analyst_Assistant_Agent.shared.contracts import AnalysisPlan, OrchestrationState


class _FakeApp:
    def __init__(self) -> None:
        self.invoked_with: dict[str, Any] | None = None

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.invoked_with = payload
        return {}


def _patch_build_app(monkeypatch, fake_app: _FakeApp) -> None:
    monkeypatch.setattr(
        "DATA_Analyst_Assistant_Agent.agents.sql.graph.build_app",
        lambda: fake_app,
    )


def test_sql_agent_uses_agent_feedback_when_present(monkeypatch) -> None:
    fake_app = _FakeApp()
    _patch_build_app(monkeypatch, fake_app)
    state = OrchestrationState(
        run_id="run_1",
        user_query="고객 단위 RFM 세그먼트를 만들어줘",
        plan=AnalysisPlan(
            goal="고객 단위 RFM",
            retry_context={
                "agent_feedback": {
                    "sql_agent": {
                        "reason": "고객 단위가 아니라 주문 단위로 생성됨",
                        "missing_evidence": ["customer_unique_id 기준 grain"],
                        "source": "semantic",
                    }
                },
                # 죽어있던 옛 키들도 같이 와도 agent_feedback을 우선해야 한다
                "message": "",
                "query": "",
            },
        ),
    )

    SQLAgent()._run_main_sql_agent(state)

    assert fake_app.invoked_with is not None
    clarification = fake_app.invoked_with["clarification_request"]
    assert "고객 단위가 아니라 주문 단위로 생성됨" in clarification
    assert "customer_unique_id 기준 grain" in clarification


def test_sql_agent_falls_back_to_legacy_retry_context_keys(monkeypatch) -> None:
    fake_app = _FakeApp()
    _patch_build_app(monkeypatch, fake_app)
    state = OrchestrationState(
        run_id="run_1",
        user_query="월별 매출 추이를 보여줘",
        plan=AnalysisPlan(
            goal="월별 매출 추이",
            retry_context={"message": "syntax error", "query": "SELECT *", "step": "call_sql_agent"},
        ),
    )

    SQLAgent()._run_main_sql_agent(state)

    assert fake_app.invoked_with is not None
    clarification = fake_app.invoked_with["clarification_request"]
    assert "syntax error" in clarification
    assert "SELECT *" in clarification


def test_sql_agent_clarification_empty_without_retry_context(monkeypatch) -> None:
    fake_app = _FakeApp()
    _patch_build_app(monkeypatch, fake_app)
    state = OrchestrationState(run_id="run_1", user_query="월별 매출 추이를 보여줘")

    SQLAgent()._run_main_sql_agent(state)

    assert fake_app.invoked_with is not None
    assert fake_app.invoked_with["clarification_request"] == ""


def test_retryable_warning_finding_is_a_limitation_not_retry_required() -> None:
    finding = SQLAgent._validation_finding(
        {
            "category": "intent_mismatch",
            "severity": "warning",
            "retryable": True,
            "detail": "Required aggregation hint NTILE was not explicitly detected in SQL.",
        }
    )

    assert finding.disposition == "limitation"
    assert finding.retryable is True

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda.nodes.tool_runner import run_tool_with_summary_and_facts


class _FailingTool:
    def invoke(self, payload: dict) -> str:
        raise RuntimeError("plot failed")


def test_tool_runner_returns_fallback_when_tool_execution_fails() -> None:
    result, facts, err = run_tool_with_summary_and_facts(
        _FailingTool(),
        lambda result_json: f"should not summarize {result_json}",
        "time",
        fallback="analysis skipped",
    )

    assert result == "analysis skipped"
    assert facts == ["analysis skipped"]
    assert err == "[time.tool] plot failed"

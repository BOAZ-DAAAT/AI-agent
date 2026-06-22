from __future__ import annotations

from data_agent_backend.agent.report.prompts import REPORT_AGENT_SYSTEM_PROMPT


def test_report_agent_prompt_contains_safety_contract() -> None:
    prompt = REPORT_AGENT_SYSTEM_PROMPT

    assert "SQL 생성" in prompt
    assert "EDA 계산" in prompt
    assert "통계 검정" in prompt
    assert "모델링" in prompt
    assert "입력에 없는 수치" in prompt
    assert "한국어" in prompt
    assert "ReportAgentOutput" in prompt

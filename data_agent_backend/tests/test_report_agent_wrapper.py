from __future__ import annotations

from data_agent_backend.agent.report.wrapper import call_create_report
from data_agent_backend.agent.schemas import ReportAgentInput, ReportAgentOutput


class FakeReportAgent:
    def __init__(self) -> None:
        self.received = None

    def invoke(self, payload):
        self.received = payload
        return {
            "structured_response": {
                "title": "카테고리 성과 리포트",
                "final_summary": "카테고리별 성과 차이가 확인되었습니다.",
                "key_findings": ["A 카테고리 주문 수가 높습니다."],
                "sections": [],
                "limitations": [],
                "next_questions": [],
                "chart_refs": [],
                "artifact_refs": [],
                "report_markdown": "# 카테고리 성과 리포트\n",
                "report_artifact_id": None,
            }
        }


def test_create_report_wrapper_invokes_subagent_with_report_input() -> None:
    agent = FakeReportAgent()
    payload = ReportAgentInput(
        user_question="카테고리별 성과를 비교해줘.",
        run_id="run_test",
        supervisor_reason="최종 보고서 작성 단계입니다.",
    )

    output = call_create_report(agent, payload)

    assert isinstance(output, ReportAgentOutput)
    assert output.title == "카테고리 성과 리포트"
    assert agent.received["messages"][0]["role"] == "user"
    assert "카테고리별 성과" in agent.received["messages"][0]["content"]

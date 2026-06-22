from __future__ import annotations

import pytest
from pydantic import ValidationError

from data_agent_backend.agent.schemas import (
    AnalysisAgentResult,
    EDAAgentResult,
    ReportAgentInput,
    ReportAgentOutput,
    SQLAgentResult,
)


def test_report_agent_input_accepts_nested_agent_results() -> None:
    payload = ReportAgentInput(
        user_question="카테고리별 성과를 비교해줘.",
        run_id="run_test",
        question_type="comparison",
        supervisor_reason="SQL, EDA, Analysis 결과를 최종 보고서로 정리해야 합니다.",
        sql_result={
            "query": "SELECT category_name, total_orders FROM category_mart",
            "target_table": "category_mart",
            "mart_design": {"grain": "category-level"},
            "artifact_refs": [{"artifact_id": "art_sql", "type": "sql_result"}],
        },
        eda_result={
            "final_summary": "카테고리별 주문 수 편차가 큽니다.",
            "stats": {"row_count": 10},
            "hypotheses": [{"id": "hyp_1", "type": "group_difference"}],
            "chart_requests": [{"intent": "카테고리별 주문 수 비교"}],
        },
        analysis_result={
            "hypothesis_results": [{"hypothesis_id": "hyp_1", "significant": True}],
            "model_results": [],
            "interpretations": ["카테고리 간 차이가 유의합니다."],
        },
    )

    assert isinstance(payload.sql_result, SQLAgentResult)
    assert isinstance(payload.eda_result, EDAAgentResult)
    assert isinstance(payload.analysis_result, AnalysisAgentResult)
    assert payload.language == "ko"


def test_report_agent_models_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ReportAgentInput(user_question="질문", run_id="run_test", unknown=True)


def test_report_agent_output_defaults_are_structured() -> None:
    output = ReportAgentOutput(
        title="카테고리 성과 리포트",
        final_summary="카테고리별 성과 차이가 확인되었습니다.",
        report_markdown="# 카테고리 성과 리포트\n",
    )

    assert output.key_findings == []
    assert output.sections == []
    assert output.limitations == []
    assert output.next_questions == []
    assert output.report_artifact_id is None

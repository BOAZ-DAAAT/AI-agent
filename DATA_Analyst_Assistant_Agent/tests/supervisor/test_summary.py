"""supervisor/summary — 노드 종류별(SQL/EDA/분석/인사이트) 의미적 구조 검증.

evidence.py가 primary_hypothesis(EDA)/method_decision·hypothesis_tests(분석)/
evidence_labels(인사이트)를 새로 읽는지, generator.py가 kind별로 다른 detail 스키마를
만들고 그 구조화 필드(LLM이 안 쓰고 근거 그대로 복사하는 것들)를 보존하는지 확인한다.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from data_agent_backend.config import BackendConfig
from data_agent_backend.models.artifacts import ArtifactType

from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.supervisor.summary import generator as generator_module
from DATA_Analyst_Assistant_Agent.supervisor.summary.evidence import read_node_evidence
from DATA_Analyst_Assistant_Agent.supervisor.summary.generator import generate_node_summary
from DATA_Analyst_Assistant_Agent.shared.numeric_verify import collect_numbers, verify_texts
from DATA_Analyst_Assistant_Agent.supervisor.summary.schemas import (
    AnalysisSummaryDetail,
    EDASummaryDetail,
    FindingSection,
    InsightSummaryDetail,
    NodeSummaryResult,
    SQLSummaryDetail,
)


@pytest.fixture()
def adapter(tmp_path) -> BackendAdapter:
    return BackendAdapter(config=BackendConfig(base_data_dir=tmp_path / ".data_agent"))


@pytest.fixture()
def runtime(adapter: BackendAdapter) -> AgentRuntime:
    return AgentRuntime(adapter=adapter)


class _FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def invoke(self, prompt: str):
        self.calls.append(prompt)
        content = self._responses.pop(0) if self._responses else self._responses[-1]
        return SimpleNamespace(content=content)


class _RaisingLLM:
    def invoke(self, prompt: str):
        raise RuntimeError("boom")


def _register(adapter, run_id, artifact_type, payload, *, kind, filename):
    content = json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict) else payload
    return adapter.register_artifact(
        run_id, artifact_type, content_text=content, filename=filename,
        created_by_tool="test.summary", metadata={"kind": kind},
    ).artifact_id


# ── evidence.py: 새로 읽어야 하는 필드들 ──

def test_eda_evidence_includes_primary_hypothesis(adapter, runtime):
    run = adapter.create_run(thread_id="thread_eda_ev")
    artifact_id = _register(
        adapter, run.run_id, ArtifactType.file,
        {
            "final_summary": "요약",
            "hypotheses": "가설 텍스트",
            "primary_hypothesis": {"target": "review_score", "feature": "delivery_days", "method": "단순선형회귀"},
            "cautions": [], "data_level": {}, "statistical_metadata": {},
        },
        kind="eda_summary", filename="eda_summary.json",
    )

    evidence = read_node_evidence([artifact_id], runtime)

    assert evidence.facts["primary_hypothesis"] == {
        "target": "review_score", "feature": "delivery_days", "method": "단순선형회귀",
    }


def test_analysis_evidence_includes_method_decision_and_hypothesis_tests(adapter, runtime):
    run = adapter.create_run(thread_id="thread_analysis_ev")
    artifact_id = _register(
        adapter, run.run_id, ArtifactType.file,
        {
            "title": "t", "executive_summary": "e", "key_findings": [], "limitations": [],
            "method_notes": [],
            "method_decision": {"selected_method": "Spearman", "rationale": "왜도가 커서"},
            "hypothesis_tests": [{"null_hypothesis": "H0", "decision": "supported"}],
        },
        kind="analysis_result", filename="analysis_result.json",
    )

    evidence = read_node_evidence([artifact_id], runtime)

    assert evidence.facts["method_decision"] == {"selected_method": "Spearman", "rationale": "왜도가 커서"}
    assert evidence.facts["hypothesis_tests"] == [{"null_hypothesis": "H0", "decision": "supported"}]


def test_insight_evidence_found_even_when_final_report_is_listed_first(adapter, runtime):
    """실사례 버그: InsightGenerator.run()은 artifact_refs를
    [report_ref(final_report), payload_ref(insight_payload), *chart_refs] 순으로 반환하므로,
    state.artifact_ids["insight"]의 0번 인덱스는 insight_payload가 아니라 final_report다.
    read_node_evidence가 0번만 보면 근거가 통째로 비어버린다(run_summary_sample 실행 중 발견)."""
    run = adapter.create_run(thread_id="thread_insight_order")
    report_id = _register(
        adapter, run.run_id, ArtifactType.report,
        "# Insight Report\n\n## Answer\n답변입니다.",
        kind="final_report", filename="final_report.md",
    )
    payload_id = _register(
        adapter, run.run_id, ArtifactType.file,
        {"answer": "실제 답변", "key_insights": [], "action_plan": [], "limitations": []},
        kind="insight_payload", filename="insight_payload.json",
    )

    evidence = read_node_evidence([report_id, payload_id], runtime)

    assert evidence.source_kind == "insight_payload"
    assert evidence.facts.get("answer") == "실제 답변"


def test_insight_evidence_includes_evidence_labels(adapter, runtime):
    run = adapter.create_run(thread_id="thread_insight_ev")
    artifact_id = _register(
        adapter, run.run_id, ArtifactType.file,
        {
            "answer": "a", "key_insights": [], "action_plan": [], "limitations": [],
            "evidence_sources": ["art_sql_1", "art_eda_1"],
            "evidence_labels": {"art_sql_1": "SQL 결과 테이블", "art_eda_1": "EDA 검증"},
        },
        kind="insight_payload", filename="insight_payload.json",
    )

    evidence = read_node_evidence([artifact_id], runtime)

    assert evidence.facts["evidence_labels"] == ["SQL 결과 테이블", "EDA 검증"]


# ── generator.py: kind별 프롬프트/파서 응답 ──

_SQL_RESPONSE = json.dumps({
    "title": "고객 RFM 마트", "subtitle": "고객 단위 집계", "background": "고객별 행동을 한 번에 보기 위함입니다.",
    "source_tables": ["orders", "order_payments"],
    "integrity_checks": ["결제 미존재 주문은 0으로 대체하였습니다."],
    "derived_columns": [{"heading": "recency_days", "body": "기준일과 최근 주문일의 차이를 계산하였습니다."}],
    "mart_grain": "customer_unique_id 당 1행",
    "mart_columns": ["customer_unique_id", "recency_days"],
    "conclusion": "고객 단위 RFM 마트를 생성하였습니다.",
    "key_finding": "고객별 RFM 지표를 집계했습니다.",
}, ensure_ascii=False)

_EDA_RESPONSE = json.dumps({
    "title": "RFM 분포 점검", "subtitle": "빈도 편중 확인", "background": "핵심 지표 분포를 점검하기 위함입니다.",
    "data_profile": "고객 단위로 집계된 데이터입니다.",
    "quality_issues": ["소규모 그룹이 다수 존재합니다."],
    "statistical_findings": [{"heading": "빈도 편중", "body": "구매 빈도가 한 쪽에 몰려 있습니다.", "chart_artifact_ids": ["chart_eda_1"]}],
    "hypotheses": ["빈도와 매출은 약한 양의 관계가 있을 것이다."],
    "charts_generated": [{"heading": "빈도 분포 차트", "body": "빈도 편중을 보여주기 위해 생성하였습니다.", "chart_artifact_ids": ["chart_eda_1"]}],
    "conclusion": "빈도 편중과 매출 롱테일을 확인하였습니다.",
    "key_finding": "빈도는 편중되고 매출은 롱테일입니다.",
}, ensure_ascii=False)

_ANALYSIS_RESPONSE = json.dumps({
    "title": "빈도-매출 관계 검정", "subtitle": "스피어만 상관검정", "background": "관계의 유의성을 확인하기 위함입니다.",
    "hypothesis_tests": [{"heading": "빈도-매출 상관", "body": "약한 양의 상관 관계가 지지되었습니다."}],
    "key_statistics": [],
    "limitations": ["표본이 한쪽에 치우쳐 있습니다."],
    "conclusion": "빈도와 매출 사이의 약한 양의 관계를 확인하였습니다.",
    "key_finding": "빈도-매출 상관이 지지되었습니다.",
}, ensure_ascii=False)

_INSIGHT_RESPONSE = json.dumps({
    "title": "고가치 저만족 고객 진단", "subtitle": "케어 우선순위 도출", "background": "고가치 고객의 이탈 위험을 파악하기 위함입니다.",
    "answer": "고가치이면서 만족도가 낮은 고객군이 존재합니다.",
    "key_insights": ["고가치 저만족 고객군이 존재합니다."],
    "action_plan": ["해당 고객군에 대한 케어 프로세스를 우선 도입합니다."],
    "limitations": ["관찰 데이터 기반으로 인과관계를 보장하지 않습니다."],
    "conclusion": "고가치 저만족 고객군에 대한 케어 우선순위를 제안합니다.",
    "key_finding": "고가치 저만족 고객군을 식별하였습니다.",
}, ensure_ascii=False)


def test_generate_sql_summary_uses_sql_detail(adapter, runtime, monkeypatch):
    run = adapter.create_run(thread_id="thread_sql_gen")
    artifact_id = _register(
        adapter, run.run_id, ArtifactType.file,
        {
            "generated_sql": "CREATE TABLE analytics.dm AS SELECT customer_unique_id FROM orders",
            "target_table": "analytics.dm", "original_question": "고객별 RFM을 보여줘",
            "target_metric": "monetary", "grain": "customer_unique_id", "business_grain": "고객 단위",
            "reasoning": "고객별 집계가 필요합니다.",
        },
        kind="sql_plan", filename="sql_plan.json",
    )
    fake_llm = _FakeLLM([_SQL_RESPONSE])
    monkeypatch.setattr(generator_module, "get_chat_model", lambda **kwargs: fake_llm)

    ref = generate_node_summary([artifact_id], runtime)
    payload = json.loads(adapter.read_artifact_text(ref.artifact_id))

    assert payload["detail"]["kind"] == "sql"
    assert payload["detail"]["source_tables"] == ["orders", "order_payments"]
    assert payload["detail"]["sql_snippet"] == "CREATE TABLE analytics.dm AS SELECT customer_unique_id FROM orders"
    assert payload["fallback_used"] is False


def test_sql_evidence_merges_plan_and_result_and_includes_preview(adapter, runtime):
    """실사례 요청: SQL 요약에 실제 마트 5행 미리보기를 담고 싶어서 sql_plan(SQL 텍스트)과
    sql_result(CSV)를 합쳐 읽게 했다 — 한쪽만 보면 코드나 미리보기 중 하나가 항상 빠진다."""
    run = adapter.create_run(thread_id="thread_sql_merge")
    plan_id = _register(
        adapter, run.run_id, ArtifactType.file,
        {"generated_sql": "SELECT customer_unique_id, monetary_total FROM dm", "target_table": "analytics.dm"},
        kind="sql_plan", filename="sql_plan.json",
    )
    result_id = adapter.register_artifact(
        run.run_id, ArtifactType.sql_result,
        content_text="customer_unique_id,monetary_total\nc1,120.5\nc2,80.0\n",
        filename="result.csv", created_by_tool="test.summary", metadata={"kind": "sql_result"},
    ).artifact_id

    evidence = read_node_evidence([plan_id, result_id], runtime)

    assert evidence.code_used == "SELECT customer_unique_id, monetary_total FROM dm"
    assert evidence.facts["target_table"] == "analytics.dm"
    assert evidence.facts["row_count"] == 2
    assert evidence.facts["preview"] == [
        {"customer_unique_id": "c1", "monetary_total": 120.5},
        {"customer_unique_id": "c2", "monetary_total": 80.0},
    ]


def test_generate_sql_summary_includes_mart_preview(adapter, runtime, monkeypatch):
    run = adapter.create_run(thread_id="thread_sql_preview")
    plan_id = _register(
        adapter, run.run_id, ArtifactType.file,
        {"generated_sql": "SELECT customer_unique_id, monetary_total FROM dm", "target_table": "analytics.dm"},
        kind="sql_plan", filename="sql_plan.json",
    )
    result_id = adapter.register_artifact(
        run.run_id, ArtifactType.sql_result,
        content_text="customer_unique_id,monetary_total\nc1,120.5\nc2,80.0\n",
        filename="result.csv", created_by_tool="test.summary", metadata={"kind": "sql_result"},
    ).artifact_id
    fake_llm = _FakeLLM([_SQL_RESPONSE])
    monkeypatch.setattr(generator_module, "get_chat_model", lambda **kwargs: fake_llm)

    ref = generate_node_summary([plan_id, result_id], runtime)
    payload = json.loads(adapter.read_artifact_text(ref.artifact_id))

    assert payload["detail"]["mart_preview"] == [
        {"customer_unique_id": "c1", "monetary_total": 120.5},
        {"customer_unique_id": "c2", "monetary_total": 80.0},
    ]
    assert payload["detail"]["sql_snippet"] == "SELECT customer_unique_id, monetary_total FROM dm"


def test_generate_eda_summary_carries_primary_hypothesis_through_untouched_by_llm(adapter, runtime, monkeypatch):
    run = adapter.create_run(thread_id="thread_eda_gen")
    primary = {"target": "review_score", "feature": "delivery_days", "method": "단순선형회귀"}
    artifact_id = _register(
        adapter, run.run_id, ArtifactType.file,
        {
            "final_summary": "요약", "hypotheses": "가설 텍스트", "primary_hypothesis": primary,
            "cautions": [], "data_level": {}, "statistical_metadata": {},
            "key_charts": [{"artifact_id": "chart_eda_1", "caption": "빈도 분포"}],
        },
        kind="eda_summary", filename="eda_summary.json",
    )
    fake_llm = _FakeLLM([_EDA_RESPONSE])
    monkeypatch.setattr(generator_module, "get_chat_model", lambda **kwargs: fake_llm)

    ref = generate_node_summary([artifact_id], runtime)
    payload = json.loads(adapter.read_artifact_text(ref.artifact_id))

    assert payload["detail"]["kind"] == "eda"
    # primary_hypothesis는 LLM 응답에 없었지만 근거에서 그대로 복사되어 나와야 한다(지어내지 않음).
    assert payload["detail"]["primary_hypothesis"] == primary
    assert payload["detail"]["statistical_findings"][0]["chart_artifact_ids"] == ["chart_eda_1"]


def test_generate_analysis_summary_carries_method_decision_through_untouched_by_llm(adapter, runtime, monkeypatch):
    run = adapter.create_run(thread_id="thread_analysis_gen")
    decision = {"selected_method": "Spearman rank-correlation", "rationale": "왜도가 커서 강건한 방법을 선택"}
    artifact_id = _register(
        adapter, run.run_id, ArtifactType.file,
        {
            "title": "t", "executive_summary": "e", "key_findings": [], "limitations": ["표본 편중"],
            "method_notes": [], "method_decision": decision,
            "hypothesis_tests": [{"null_hypothesis": "H0", "decision": "supported"}],
        },
        kind="analysis_result", filename="analysis_result.json",
    )
    fake_llm = _FakeLLM([_ANALYSIS_RESPONSE])
    monkeypatch.setattr(generator_module, "get_chat_model", lambda **kwargs: fake_llm)

    ref = generate_node_summary([artifact_id], runtime)
    payload = json.loads(adapter.read_artifact_text(ref.artifact_id))

    assert payload["detail"]["kind"] == "analysis"
    assert payload["detail"]["method_decision"] == decision
    assert payload["detail"]["hypothesis_tests"][0]["heading"] == "빈도-매출 상관"


def test_generate_insight_summary_uses_evidence_labels(adapter, runtime, monkeypatch):
    run = adapter.create_run(thread_id="thread_insight_gen")
    artifact_id = _register(
        adapter, run.run_id, ArtifactType.file,
        {
            "answer": "a", "key_insights": [], "action_plan": [], "limitations": [],
            "evidence_sources": ["art_sql_1"], "evidence_labels": {"art_sql_1": "SQL 결과 테이블"},
        },
        kind="insight_payload", filename="insight_payload.json",
    )
    fake_llm = _FakeLLM([_INSIGHT_RESPONSE])
    monkeypatch.setattr(generator_module, "get_chat_model", lambda **kwargs: fake_llm)

    ref = generate_node_summary([artifact_id], runtime)
    payload = json.loads(adapter.read_artifact_text(ref.artifact_id))

    assert payload["detail"]["kind"] == "insight"
    assert payload["detail"]["evidence_sources"] == ["SQL 결과 테이블"]


def test_collect_texts_excludes_chart_artifact_ids_from_numeric_verification():
    """run-summary_sample 실사례: artifact_id 안 숫자 조각("art_a374da05...")이 근거없는
    숫자로 오탐되어 EDA/인사이트 서머리가 계속 폴백되던 버그의 회귀 테스트."""
    result = NodeSummaryResult(
        title="t", subtitle="s", background="b", conclusion="c", key_finding="k",
        source_kind="eda_summary",
        detail=EDASummaryDetail(
            statistical_findings=[FindingSection(
                heading="관계 분석", body="상관계수는 -0.367입니다.",
                chart_artifact_ids=["art_a374da05e2394a29b4401fa83de851f0"],
            )],
        ),
    )

    texts = generator_module._collect_texts(result)

    assert "art_a374da05e2394a29b4401fa83de851f0" not in texts
    assert not any("374da05" in t for t in texts)

    numbers = collect_numbers({"pearson_r": -0.367})
    corpus = json.dumps({"pearson_r": -0.367}, ensure_ascii=False)
    ok, missing = verify_texts(texts, numbers, corpus)
    assert ok, missing


def test_generate_summary_falls_back_per_kind_when_llm_fails(adapter, runtime, monkeypatch):
    run = adapter.create_run(thread_id="thread_fallback")
    artifact_id = _register(
        adapter, run.run_id, ArtifactType.file,
        {"final_summary": "요약", "hypotheses": "", "primary_hypothesis": {}, "cautions": [],
         "data_level": {}, "statistical_metadata": {}},
        kind="eda_summary", filename="eda_summary.json",
    )
    monkeypatch.setattr(generator_module, "get_chat_model", lambda **kwargs: _RaisingLLM())

    ref = generate_node_summary([artifact_id], runtime)
    payload = json.loads(adapter.read_artifact_text(ref.artifact_id))

    assert payload["fallback_used"] is True
    assert payload["detail"]["kind"] == "eda"


def test_generate_node_summary_uses_cache_on_second_call(adapter, runtime, monkeypatch):
    run = adapter.create_run(thread_id="thread_cache")
    artifact_id = _register(
        adapter, run.run_id, ArtifactType.file,
        {
            "generated_sql": "SELECT 1", "target_table": "t", "original_question": "q",
            "target_metric": "m", "grain": "g", "business_grain": "bg", "reasoning": "r",
        },
        kind="sql_plan", filename="sql_plan.json",
    )
    fake_llm = _FakeLLM([_SQL_RESPONSE])
    monkeypatch.setattr(generator_module, "get_chat_model", lambda **kwargs: fake_llm)

    ref1 = generate_node_summary([artifact_id], runtime)
    ref2 = generate_node_summary([artifact_id], runtime)

    assert ref1.artifact_id == ref2.artifact_id
    assert len(fake_llm.calls) == 1


def test_generate_node_summary_requires_artifact_ids(runtime):
    with pytest.raises(ValueError):
        generate_node_summary([], runtime)

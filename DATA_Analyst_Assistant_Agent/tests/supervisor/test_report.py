from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from data_agent_backend.config import BackendConfig
from data_agent_backend.models.artifacts import ArtifactType

from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.supervisor.report import generator as generator_module
from DATA_Analyst_Assistant_Agent.supervisor.report.evidence import read_path_evidence
from DATA_Analyst_Assistant_Agent.supervisor.report.generator import generate_report


@pytest.fixture()
def adapter(tmp_path) -> BackendAdapter:
    return BackendAdapter(config=BackendConfig(base_data_dir=tmp_path / ".data_agent"))


@pytest.fixture()
def runtime(adapter: BackendAdapter) -> AgentRuntime:
    return AgentRuntime(adapter=adapter)


class _FakeLLM:
    """큐에 넣어둔 응답을 순서대로 돌려준다(마지막 응답은 반복)."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def invoke(self, prompt: str):
        self.calls.append(prompt)
        content = self._responses.pop(0) if self._responses else self._responses[-1]
        return SimpleNamespace(content=content)


def _register(adapter, run_id, artifact_type, payload, *, kind, filename):
    if isinstance(payload, dict):
        content = json.dumps(payload, ensure_ascii=False)
    else:
        content = payload
    return adapter.register_artifact(
        run_id,
        artifact_type,
        content_text=content,
        filename=filename,
        created_by_tool="test.report",
        metadata={"kind": kind},
    ).artifact_id


def _seed_full_path(adapter: BackendAdapter, run_id: str) -> dict[str, str]:
    sql_plan_id = _register(
        adapter, run_id, ArtifactType.file,
        {
            "sql_draft": {
                "sql": "SELECT category, SUM(sales_amount) AS total FROM orders GROUP BY category",
                "target_table": "analytics.orders_mart",
                "business_grain": "카테고리별 합계",
                "reasoning": "카테고리 비교가 필요해 그룹화했다",
            },
            "plan": {
                "original_question": "카테고리별 매출을 비교해줘",
                "target_metric": "sales_amount",
                "grain": "category",
            },
        },
        kind="sql_plan", filename="sql_plan.json",
    )
    sql_result_id = _register(
        adapter, run_id, ArtifactType.sql_result,
        "category,total\n가전,1000000\n뷰티,500000\n",
        kind="sql_result", filename="result.csv",
    )
    eda_id = _register(
        adapter, run_id, ArtifactType.file,
        {
            "final_summary": "가전 카테고리의 매출 분산이 가장 크고 이상치 후보가 존재합니다.",
            "hypotheses": "이상치가 평균을 왜곡시켰을 수 있다",
            "cautions": ["6월 주문 중 이상치 후보 존재"],
            "data_level": {"row_count": 2},
            "statistical_metadata": {"mean": 750000, "adhoc_analysis": {"code": "df.groupby('category').sum()"}},
            "key_charts": [{"artifact_id": "chart_eda_1", "caption": "카테고리별 분포"}],
        },
        kind="eda_summary", filename="eda_summary.json",
    )
    analysis_id = _register(
        adapter, run_id, ArtifactType.file,
        {
            "title": "카테고리별 매출 원인 분석",
            "executive_summary": "이상치를 제외하면 매출 차이는 크지 않습니다.",
            "key_findings": ["이상치 제외 후 두 카테고리 평균 매출 차이는 5% 이내"],
            "limitations": ["표본 크기가 작음"],
            "method_notes": ["이상치 제거 후 재계산"],
            "generated_code": "",
        },
        kind="analysis_result", filename="analysis_result.json",
    )
    insight_id = _register(
        adapter, run_id, ArtifactType.file,
        {
            "answer": "가전이 매출 1위지만 이상치 영향이 커서 실질 차이는 작습니다.",
            "key_insights": ["이상치 제외 시 카테고리 간 매출 차이는 크지 않음"],
            "action_plan": ["이상치 주문 검토 프로세스 도입"],
            "limitations": ["단기간 데이터"],
            "charts": [{"artifact_id": "chart_insight_1", "title": "카테고리별 매출 비교"}],
        },
        kind="insight_payload", filename="insight_payload.json",
    )
    return {
        "sql_plan": sql_plan_id,
        "sql_result": sql_result_id,
        "eda": eda_id,
        "analysis": analysis_id,
        "insight": insight_id,
    }


_VALID_RESPONSE = json.dumps({
    "title": "카테고리별 매출 원인 분석 여정",
    "executive_summary": "가전 카테고리의 이상치를 제외하면 매출 차이는 크지 않은 것으로 나타났습니다.",
    "background_and_question": "사용자는 카테고리별 매출 현황을 파악하고자 했습니다.",
    "methodology_narrative": "집계된 매출 데이터를 바탕으로 분포와 이상치를 함께 검토했습니다.",
    "key_findings": [
        {"heading": "이상치 영향", "body": "이상치를 제외한 재검토 결과 두 카테고리의 매출 차이는 크지 않았습니다.",
         "chart_artifact_ids": ["chart_eda_1"]},
    ],
    "limitations": ["표본 크기가 작습니다."],
    "conclusion_and_recommendations": "이상치 검토 프로세스 도입을 제안합니다.",
    "key_finding": "이상치 제외 시 카테고리 간 매출 차이는 크지 않음",
}, ensure_ascii=False)


# ── evidence.py ──

def test_read_path_evidence_groups_all_four_stages(adapter, runtime):
    run = adapter.create_run(thread_id="thread_report")
    ids = _seed_full_path(adapter, run.run_id)

    evidence = read_path_evidence(list(ids.values()), runtime)

    assert evidence.present_stage_names == ["sql", "eda", "analysis", "insight"]
    assert evidence.user_question == "카테고리별 매출을 비교해줘"
    sql_stage = next(s for s in evidence.stages if s.stage == "sql")
    assert "SELECT category" in sql_stage.code_used
    assert sql_stage.facts["result_preview"]["row_count"] == 2
    eda_stage = next(s for s in evidence.stages if s.stage == "eda")
    assert eda_stage.charts[0].artifact_id == "chart_eda_1"
    insight_stage = next(s for s in evidence.stages if s.stage == "insight")
    assert insight_stage.charts[0].caption == "카테고리별 매출 비교"


def test_read_path_evidence_skips_missing_eda_stage(adapter, runtime):
    run = adapter.create_run(thread_id="thread_report_no_eda")
    ids = _seed_full_path(adapter, run.run_id)
    without_eda = [v for k, v in ids.items() if k != "eda"]

    evidence = read_path_evidence(without_eda, runtime)

    assert evidence.present_stage_names == ["sql", "analysis", "insight"]
    assert "eda" not in evidence.present_stage_names


# ── generator.py ──

def test_generate_report_success(adapter, runtime, monkeypatch):
    run = adapter.create_run(thread_id="thread_report_success")
    ids = _seed_full_path(adapter, run.run_id)
    fake_llm = _FakeLLM([_VALID_RESPONSE])
    monkeypatch.setattr(generator_module, "get_chat_model", lambda **kwargs: fake_llm)

    ref = generate_report(list(ids.values()), runtime)

    record = adapter.get_artifact(ref.artifact_id)
    payload = json.loads(adapter.read_artifact_text(ref.artifact_id))
    assert record.metadata["kind"] == "report"
    assert set(record.parent_ids) == set(ids.values())
    assert payload["fallback_used"] is False
    assert payload["included_stages"] == ["sql", "eda", "analysis", "insight"]
    assert len(fake_llm.calls) == 1


def test_generate_report_uses_cache_on_second_call(adapter, runtime, monkeypatch):
    run = adapter.create_run(thread_id="thread_report_cache")
    ids = _seed_full_path(adapter, run.run_id)
    fake_llm = _FakeLLM([_VALID_RESPONSE])
    monkeypatch.setattr(generator_module, "get_chat_model", lambda **kwargs: fake_llm)

    first = generate_report(list(ids.values()), runtime)
    second = generate_report(list(ids.values()), runtime)

    assert first.artifact_id == second.artifact_id
    assert len(fake_llm.calls) == 1  # 두 번째 호출은 캐시로 처리되어 LLM을 다시 안 부른다


def test_generate_report_retries_on_listing_style(adapter, runtime, monkeypatch):
    run = adapter.create_run(thread_id="thread_report_listing")
    ids = _seed_full_path(adapter, run.run_id)
    listing_response = json.dumps({
        "title": "카테고리별 매출 분석 여정",
        "executive_summary": "가전과 뷰티 카테고리의 매출을 비교했습니다.",
        "background_and_question": "SQL 단계에서는 카테고리별 매출 합계를 구했습니다.",
        "methodology_narrative": "EDA 단계에서는 분포와 이상치를 확인했습니다.",
        "key_findings": [{"heading": "매출 비교", "body": "가전이 가장 높았습니다.", "chart_artifact_ids": []}],
        "limitations": [],
        "conclusion_and_recommendations": "이상치 검토가 필요합니다.",
        "key_finding": "가전이 매출 1위",
    }, ensure_ascii=False)
    fake_llm = _FakeLLM([listing_response, _VALID_RESPONSE])
    monkeypatch.setattr(generator_module, "get_chat_model", lambda **kwargs: fake_llm)

    ref = generate_report(list(ids.values()), runtime)

    payload = json.loads(adapter.read_artifact_text(ref.artifact_id))
    assert len(fake_llm.calls) == 2  # 첫 응답은 나열식이라 거부되고 재시도됨
    assert payload["fallback_used"] is False
    assert payload["title"] == "카테고리별 매출 원인 분석 여정"  # 두 번째(정상) 응답이 채택됨


def test_generate_report_retries_on_invented_number(adapter, runtime, monkeypatch):
    run = adapter.create_run(thread_id="thread_report_hallucination")
    ids = _seed_full_path(adapter, run.run_id)
    hallucinated_response = json.dumps({
        "title": "카테고리별 매출 원인 분석 여정",
        "executive_summary": "이번 분석 결과 매출이 42.7% 증가한 것으로 나타났습니다.",
        "background_and_question": "사용자는 카테고리별 매출 현황을 파악하고자 했습니다.",
        "methodology_narrative": "집계된 매출 데이터를 바탕으로 분포와 이상치를 함께 검토했습니다.",
        "key_findings": [{"heading": "매출 증가", "body": "42.7%의 성장을 보였습니다.", "chart_artifact_ids": []}],
        "limitations": [],
        "conclusion_and_recommendations": "성장세를 유지해야 합니다.",
        "key_finding": "매출 42.7% 증가",
    }, ensure_ascii=False)
    fake_llm = _FakeLLM([hallucinated_response, _VALID_RESPONSE])
    monkeypatch.setattr(generator_module, "get_chat_model", lambda **kwargs: fake_llm)

    ref = generate_report(list(ids.values()), runtime)

    payload = json.loads(adapter.read_artifact_text(ref.artifact_id))
    assert len(fake_llm.calls) == 2  # 근거에 없는 42.7%가 걸려서 재시도됨
    assert payload["fallback_used"] is False
    assert payload["title"] == "카테고리별 매출 원인 분석 여정"
    assert "42.7" not in json.dumps(payload, ensure_ascii=False)


def test_generate_report_falls_back_when_llm_fails(adapter, runtime, monkeypatch):
    run = adapter.create_run(thread_id="thread_report_fallback")
    ids = _seed_full_path(adapter, run.run_id)

    class _RaisingLLM:
        def invoke(self, prompt: str):
            raise RuntimeError("boom")

    monkeypatch.setattr(generator_module, "get_chat_model", lambda **kwargs: _RaisingLLM())

    ref = generate_report(list(ids.values()), runtime)

    payload = json.loads(adapter.read_artifact_text(ref.artifact_id))
    assert payload["fallback_used"] is True
    assert payload["included_stages"] == ["sql", "eda", "analysis", "insight"]
    assert len(payload["key_findings"]) > 0


def test_generate_report_works_without_eda_stage(adapter, runtime, monkeypatch):
    run = adapter.create_run(thread_id="thread_report_no_eda_gen")
    ids = _seed_full_path(adapter, run.run_id)
    without_eda = [v for k, v in ids.items() if k != "eda"]
    fake_llm = _FakeLLM([_VALID_RESPONSE])
    monkeypatch.setattr(generator_module, "get_chat_model", lambda **kwargs: fake_llm)

    ref = generate_report(without_eda, runtime)

    payload = json.loads(adapter.read_artifact_text(ref.artifact_id))
    assert payload["included_stages"] == ["sql", "analysis", "insight"]
    assert payload["fallback_used"] is False


def test_generate_report_requires_artifact_ids(runtime):
    with pytest.raises(ValueError):
        generate_report([], runtime)

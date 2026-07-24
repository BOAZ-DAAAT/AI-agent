"""Insight Agent 테스트 — 전부 FakeLLM/FakeAdapter (실제 LLM·백엔드 호출 0, 토큰 0).

검증 대상: 증거팩 조립 / compute·chart 도구(게이트·범위·렌더) /
bounded ReAct 루프(성공·빈 answer 피드백·품질심사·폴백) / InsightGenerator.run 아티팩트 등록.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pandas as pd
import pytest

from data_agent_backend.models.artifacts import ArtifactRef, ArtifactType

from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.supervisor.insight.evidence import EvidencePack, build_evidence_pack
from DATA_Analyst_Assistant_Agent.supervisor.insight.loop import run_insight_loop
from DATA_Analyst_Assistant_Agent.supervisor.insight.tools import run_chart, run_compute, run_look
from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState

# ─────────────────────────────
# 공용 페이크
# ─────────────────────────────
_CSV = "category,total_sales\ntoys,500.0\ntoys,400.0\nauto,300.0\npet,100.0\n"
_EDA_JSON = json.dumps({
    "final_summary": "toys 카테고리 중심의 매출 집중 구조입니다.",
    "cautions": [{"code": "LOW_N", "severity": "low", "message_ko": "표본 작음"}],
    "statistical_metadata": {"row_count": 4},
})
_ANALYSIS_JSON = json.dumps({
    "method_summary": "카테고리별 매출 비교 분석",
    "key_findings": ["toys가 매출 1위(900.0)"],
    "limitations": ["관찰 데이터"],
})


class FakeLLM:
    """호출마다 준비된 응답을 순서대로 돌려주는 가짜 LLM."""

    def __init__(self, *responses: str):
        self._seq = list(responses)
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1
        content = self._seq.pop(0) if self._seq else "{}"
        return SimpleNamespace(content=content)


class FakeAdapter:
    """증거 읽기 + 아티팩트 등록만 흉내낸다 (content_bytes 지원 여부 선택 가능)."""

    def __init__(self, artifacts: dict | None = None, support_bytes: bool = True):
        self._artifacts = artifacts or {}              # id → (ArtifactType, text)
        self._support_bytes = support_bytes
        self.registered: list[dict] = []
        self._n = 0

    def get_artifact(self, artifact_id: str):
        art_type, _ = self._artifacts[artifact_id]
        return SimpleNamespace(artifact_id=artifact_id, type=art_type, metadata={})

    def read_artifact_text(self, artifact_id: str) -> str:
        return self._artifacts[artifact_id][1]

    def register_artifact(self, run_id, artifact_type, **kwargs):
        if "content_bytes" in kwargs and not self._support_bytes:
            raise TypeError("content_bytes not supported")   # 실제 adapter 현재 동작 재현
        self._n += 1
        self.registered.append({"type": str(artifact_type), **{k: v for k, v in kwargs.items()
                                                               if k in ("filename", "metadata")}})
        return ArtifactRef(artifact_id=f"art-{self._n}", type=ArtifactType(artifact_type))


def _state(**artifact_ids) -> OrchestrationState:
    return OrchestrationState(run_id="r1", user_query="카테고리별 매출 상위는?",
                              artifact_ids=artifact_ids or {}, route_kind="comprehensive")


def _runtime(adapter: FakeAdapter) -> AgentRuntime:
    return AgentRuntime(adapter=adapter)


def _pack(with_eda: bool = False) -> EvidencePack:
    df = pd.read_csv(__import__("io").StringIO(_CSV))
    from DATA_Analyst_Assistant_Agent.supervisor.insight.evidence import _summarize_table
    eda = {"final_summary": "toys 중심 매출 집중 구조입니다."} if with_eda else {}
    return EvidencePack(user_question="카테고리별 매출 상위는?", route_kind="simple",
                        df=df, table_summary=_summarize_table(df), eda=eda)


# ─────────────────────────────
# 증거팩
# ─────────────────────────────
def test_evidence_pack_simple_csv_only():
    adapter = FakeAdapter({"s1": (ArtifactType.sql_result, _CSV)})
    pack = build_evidence_pack(_state(sql_agent=["s1"]), _runtime(adapter))
    assert pack.df is not None and len(pack.df) == 4
    assert pack.table_summary["row_count"] == 4
    assert pack.eda == {} and pack.analysis == {}
    assert pack.source_artifact_ids == ["s1"]


def test_evidence_pack_picks_largest_sql_result():
    # mart 경로: '마트 생성 완료' 1행 상태 CSV 가 먼저 와도 실제 데이터(행 많은 쪽)를 골라야 함
    status_csv = "col_1,col_2\n마트 생성 완료,analytics.mart\n"
    adapter = FakeAdapter({"s0": (ArtifactType.sql_result, status_csv),
                           "s1": (ArtifactType.sql_result, _CSV)})
    pack = build_evidence_pack(_state(sql_agent=["s0", "s1"]), _runtime(adapter))
    assert list(pack.df.columns) == ["category", "total_sales"]
    assert pack.table_summary["row_count"] == 4


def test_evidence_pack_comprehensive():
    adapter = FakeAdapter({
        "s1": (ArtifactType.sql_result, _CSV),
        "e1": (ArtifactType.data_profile, _EDA_JSON),
        "a1": (ArtifactType.file, _ANALYSIS_JSON),
    })
    pack = build_evidence_pack(
        _state(sql_agent=["s1"], eda_agent=["e1"], analysis_agent=["a1"]), _runtime(adapter))
    assert pack.eda["final_summary"].startswith("toys")
    assert pack.analysis["key_findings"] == ["toys가 매출 1위(900.0)"]
    assert set(pack.source_artifact_ids) == {"s1", "e1", "a1"}


# ─────────────────────────────
# 도구: compute / chart / look
# ─────────────────────────────
def test_evidence_pack_keeps_raw_analysis_for_lookup():
    analysis_payload = {
        "method_summary": "compact summary",
        "key_findings": ["finding"],
        "evidence": [{"statistics": {"p_value": 0.03}}],
        "debug_artifact_id": "ad1",
    }
    debug_payload = {
        "terminal_reason": "validated_result",
        "generated_code": "result = {'summary': 'ok'}",
        "raw_statistics": [{"p_value": 0.03}],
    }
    adapter = FakeAdapter({
        "s1": (ArtifactType.sql_result, _CSV),
        "a1": (ArtifactType.file, json.dumps(analysis_payload)),
        "ad1": (ArtifactType.file, json.dumps(debug_payload)),
    })

    pack = build_evidence_pack(_state(sql_agent=["s1"], analysis_agent=["a1", "ad1"]), _runtime(adapter))

    assert pack.analysis == {
        "method_summary": "compact summary",
        "key_findings": ["finding"],
        "limitations": [],
        "data_quality_notes": [],
    }
    assert pack.analysis_raw["evidence"][0]["statistics"]["p_value"] == 0.03
    assert pack.analysis_debug["generated_code"].startswith("result =")
    assert set(pack.source_artifact_ids) == {"s1", "a1", "ad1"}


def test_look_can_read_raw_analysis_and_debug_paths():
    pack = EvidencePack(
        user_question="q",
        route_kind="simple",
        analysis={"method_summary": "compact"},
        analysis_raw={"evidence": [{"statistics": {"effect_size": 1.2}}]},
        analysis_debug={"raw_statistics": [{"p_value": 0.03}]},
        raw_artifact_ids={"analysis": "a1", "analysis_debug": "ad1"},
    )

    raw = run_look(pack, {"target": "analysis_raw", "path": "evidence.0.statistics.effect_size"})
    debug = run_look(pack, {"target": "analysis_debug", "path": "raw_statistics.0.p_value"})

    assert raw["ok"] and raw["excerpt"] == 1.2 and raw["artifact_id"] == "a1"
    assert debug["ok"] and debug["excerpt"] == 0.03 and debug["artifact_id"] == "ad1"


def test_compute_success():
    out = run_compute(_pack(), {"expression": "df.groupby('category')['total_sales'].sum().nlargest(3)"})
    assert out["ok"] and out["result"]["toys"] == 900.0


def test_compute_gate_rejects_query():
    out = run_compute(_pack(), {"expression": 'df.query("total_sales > 0")'})
    assert not out["ok"] and "gate_rejected" in out["error"]


def test_compute_scope_rejects_regression():
    out = run_compute(_pack(), {"expression": "np.polyfit(df['total_sales'], df['total_sales'], 1)"})
    assert not out["ok"] and "scope_rejected" in out["error"]


def test_compute_rejects_fabricated_series():
    # pd.Series([...]) 로 임의 숫자를 만들어 '증거'로 편입시키는 구멍 차단 (코덱스 리뷰 반영)
    out = run_compute(_pack(), {"expression": "pd.Series([7777.7])"})
    assert not out["ok"] and "scope_rejected" in out["error"]


def test_compute_rejects_df_free_expression():
    out = run_compute(_pack(), {"expression": "np.sqrt(2) * 100"})
    assert not out["ok"] and "df" in out["error"]


def test_compute_without_table():
    pack = EvidencePack(user_question="q", route_kind="simple", df=None)
    out = run_compute(pack, {"expression": "df.mean()"})
    assert not out["ok"] and "no_table" in out["error"]


def test_chart_bar_renders_png(tmp_path):
    out = run_chart(_pack(), {"expression": "df.groupby('category')['total_sales'].sum()",
                              "kind": "bar", "title": "카테고리별 매출"}, str(tmp_path))
    assert out["ok"], out.get("error")
    assert os.path.exists(out["local_path"])


def test_chart_table_renders_png(tmp_path):
    out = run_chart(_pack(), {"expression": "df.groupby('category')['total_sales'].agg(['sum','mean'])",
                              "kind": "table", "title": "성능 요약"}, str(tmp_path))
    assert out["ok"], out.get("error")
    assert os.path.exists(out["local_path"])


def test_chart_y_column_selects_metric(tmp_path):
    # 제목-축 불일치 방지: LLM이 y로 지정한 컬럼을 그린다. 없는 컬럼이면 명확히 거부.
    expr = "df.groupby('category')['total_sales'].agg(['count','sum'])"
    ok = run_chart(_pack(), {"expression": expr, "kind": "bar", "title": "합계", "y": "sum"}, str(tmp_path))
    assert ok["ok"], ok.get("error")
    bad = run_chart(_pack(), {"expression": expr, "kind": "bar", "title": "x", "y": "amount"}, str(tmp_path))
    assert not bad["ok"] and "y_column_not_found" in bad["error"]


def test_chart_grouped_bar_renders(tmp_path):
    # 다지표 '특성' 비교용 grouped_bar — 여러 y 컬럼을 나란히
    out = run_chart(_pack(), {"expression": "df.groupby('category')['total_sales'].agg(['mean','sum'])",
                              "kind": "grouped_bar", "title": "카테고리 특성",
                              "y": ["mean", "sum"]}, str(tmp_path))
    assert out["ok"], out.get("error")
    assert os.path.exists(out["local_path"])


def test_fmt_num_no_scientific_notation():
    from DATA_Analyst_Assistant_Agent.supervisor.insight.tools import _fmt_num
    assert _fmt_num(25236.0) == "25,236"               # 2.524e+04 방지
    assert _fmt_num(163.567) == "163.57"
    assert _fmt_num(0.845) == "0.845"


def test_chart_scalar_not_chartable(tmp_path):
    out = run_chart(_pack(), {"expression": "df['total_sales'].mean()",
                              "kind": "bar", "title": "x"}, str(tmp_path))
    assert not out["ok"] and "not_chartable" in out["error"]


def test_look_eda_dotted_path():
    pack = _pack(with_eda=True)
    out = run_look(pack, {"target": "eda", "path": "final_summary"})
    assert out["ok"] and "toys" in out["excerpt"]


def test_look_empty_analysis_tells_agent_to_stop():
    # 없는 증거를 반복해서 들여다보는 배회 방지 — 빈 payload 는 명확한 에러로 응답
    out = run_look(_pack(), {"target": "analysis"})
    assert not out["ok"] and "없다" in out["error"]


# ─────────────────────────────
# ReAct 루프
# ─────────────────────────────
_FINISH_GOOD = json.dumps({"tool": "finish", "reason": "충분", "args": {
    "answer": "toys 카테고리가 합계 900.0으로 매출 1위입니다.",
    "key_insights": ["상위 카테고리에 매출 집중"], "action_plan": [], "limitations": []}})
_FINISH_BAD = json.dumps({"tool": "finish", "reason": "성급", "args": {
    "answer": "매출이 7777.7로 사상 최대입니다."}})
_COMPUTE = json.dumps({"tool": "compute", "reason": "합계 계산", "args": {
    "expression": "df.groupby('category')['total_sales'].sum().nlargest(3)"}})
_CHART = json.dumps({"tool": "chart", "reason": "답 증명", "args": {
    "expression": "df.groupby('category')['total_sales'].sum().nlargest(3)",
    "kind": "bar", "title": "카테고리별 매출 상위"}})


def test_loop_happy_path(tmp_path):
    llm = FakeLLM(_COMPUTE, _CHART, _FINISH_GOOD)
    result = run_insight_loop(_pack(), llm=llm, out_dir=str(tmp_path))
    assert not result.fallback_used
    assert "900.0" in result.answer
    assert len(result.charts) == 1 and os.path.exists(result.charts[0].local_path)
    assert result.rounds == 3
    # 숫자 게이트 통과 후 내부 validator(품질 심사)까지 자동으로 걸린다
    assert [s["tool"] for s in result.steps] == ["compute", "chart", "finish", "validate"]


def test_loop_accepts_ungrounded_numbers_without_verification(tmp_path):
    # 숫자 검증 게이트를 없앴다(2026-07-23) — 근거에 없는 숫자를 써도 즉시 finish 로 수용된다.
    llm = FakeLLM(_COMPUTE, _FINISH_BAD)
    result = run_insight_loop(_pack(), llm=llm, out_dir=str(tmp_path))
    assert not result.fallback_used and "7777.7" in result.answer
    finishes = [s for s in result.steps if s["tool"] == "finish"]
    assert len(finishes) == 1 and finishes[0]["ok"]


def test_loop_repeated_empty_answer_falls_back_to_grounded_summary(tmp_path):
    empty_finish = json.dumps({"tool": "finish", "reason": "빈 답", "args": {"answer": ""}})
    llm = FakeLLM(empty_finish, empty_finish, empty_finish)
    result = run_insight_loop(_pack(with_eda=True), llm=llm, out_dir=str(tmp_path))
    assert result.fallback_used
    assert "toys 중심" in result.answer                 # 검증된 상류 요약으로 폴백(지어낸 숫자 없음)
    assert result.limitations


def test_loop_parse_error_consumes_round(tmp_path):
    llm = FakeLLM("이건 JSON이 아님", _FINISH_GOOD.replace("900.0", "500.0"))
    result = run_insight_loop(_pack(), llm=llm, out_dir=str(tmp_path))
    assert not result.fallback_used and "500.0" in result.answer


def test_loop_repeated_identical_call_is_blocked(tmp_path):
    # 같은 look 을 반복하면 실행하지 않고 넛지만 준다(라운드 소진 배회 방지)
    look = json.dumps({"tool": "look", "reason": "테이블", "args": {"target": "table"}})
    llm = FakeLLM(look, look, _FINISH_GOOD.replace("900.0", "500.0"))
    result = run_insight_loop(_pack(), llm=llm, out_dir=str(tmp_path))
    assert not result.fallback_used
    repeats = [s for s in result.steps if s.get("note") == "repeat_call"]
    assert len(repeats) == 1


def test_loop_validator_retry_then_accept(tmp_path):
    # 숫자는 맞지만 품질 미달(예: 동문서답) → validator 가 피드백 → 재작성 finish 는 수용(재심사 1회 제한)
    finish1 = _FINISH_GOOD.replace("900.0", "500.0")
    retry_verdict = json.dumps({"verdict": "retry", "feedback": "질문에 더 직접적으로 답하라"})
    finish2 = json.dumps({"tool": "finish", "reason": "재작성", "args": {
        "answer": "직답: toys가 최대 매출 500.0입니다."}})
    result = run_insight_loop(_pack(), llm=FakeLLM(finish1, retry_verdict, finish2), out_dir=str(tmp_path))
    assert not result.fallback_used and result.answer.startswith("직답")
    v = [s for s in result.steps if s["tool"] == "validate"]
    assert len(v) == 1 and not v[0]["ok"] and "직접적" in v[0]["note"]


def test_loop_fail_streak_exits_early(tmp_path):
    # 연속 실패 3번(거부→반복→반복)이면 남은 라운드를 안 태우고 조기 폴백(토큰 낭비 방지)
    bad = json.dumps({"tool": "compute", "reason": "", "args": {"expression": 'df.query("a>0")'}})
    llm = FakeLLM(bad, bad, bad, bad, bad, bad, bad, bad)
    result = run_insight_loop(_pack(with_eda=True), llm=llm, out_dir=str(tmp_path))
    assert result.fallback_used
    assert llm.calls == 3                              # 8라운드 다 안 돌고 3번에 끊음


def test_loop_action_plan_gets_forced_caveat(tmp_path):
    finish = json.dumps({"tool": "finish", "reason": "", "args": {
        "answer": "toys가 최대 매출 500.0을 기록했습니다.",
        "action_plan": ["toys 재고를 우선 점검하세요."]}})
    result = run_insight_loop(_pack(), llm=FakeLLM(finish), out_dir=str(tmp_path))
    assert result.action_plan
    assert any("인과 검증" in s for s in result.limitations)   # 권고엔 caveat 강제 부착


# ─────────────────────────────
# InsightGenerator.run (end-to-end, 페이크)
# ─────────────────────────────
def _run_agent(monkeypatch, tmp_path, adapter):
    import DATA_Analyst_Assistant_Agent.supervisor.insight.loop as L
    from DATA_Analyst_Assistant_Agent.supervisor.insight import InsightGenerator
    monkeypatch.setenv("INSIGHT_CHART_DIR", str(tmp_path))
    monkeypatch.setattr(L, "get_chat_model", lambda *a, **k: FakeLLM(_COMPUTE, _CHART, _FINISH_GOOD))
    return InsightGenerator().run(_state(sql_agent=["s1"]), _runtime(adapter))


def test_markdown_embeds_chart_and_labels_sources():
    from DATA_Analyst_Assistant_Agent.supervisor.insight.agent import _build_markdown
    payload = {"answer": "답", "key_insights": [], "action_plan": [], "limitations": [],
               "charts": [{"title": "차트", "filename": "c.png", "supports": "answer",
                           "local_path": "/tmp/insight_charts/c.png", "kind": "bar", "artifact_id": None}],
               "user_question": "질문", "evidence_sources": ["a1"],
               "evidence_labels": {"a1": "SQL 결과 테이블"}, "fallback_used": False}
    md = _build_markdown(payload)
    assert "![차트](insight_charts/c.png)" in md        # 이미지 임베드
    assert "SQL 결과 테이블 (`a1`)" in md               # 사람이 읽는 출처 라벨


def test_agent_run_registers_report_payload_and_chart(monkeypatch, tmp_path):
    adapter = FakeAdapter({"s1": (ArtifactType.sql_result, _CSV)})
    envelope = _run_agent(monkeypatch, tmp_path, adapter)
    assert envelope.agent_name == "insight"
    assert "900.0" in envelope.summary
    filenames = [r["filename"] for r in adapter.registered]
    assert "final_report.md" in filenames and "insight_payload.json" in filenames
    assert len(envelope.artifact_refs) == 3            # report + payload + chart
    assert not envelope.validation.has_errors


def test_agent_run_chart_guard_when_bytes_unsupported(monkeypatch, tmp_path):
    adapter = FakeAdapter({"s1": (ArtifactType.sql_result, _CSV)}, support_bytes=False)
    envelope = _run_agent(monkeypatch, tmp_path, adapter)   # 차트 등록 실패해도 안 죽어야 함
    assert len(envelope.artifact_refs) == 2            # report + payload (chart 폴백)
    assert not envelope.validation.has_errors


def test_agent_run_without_any_evidence_flags_error(monkeypatch, tmp_path):
    import DATA_Analyst_Assistant_Agent.supervisor.insight.loop as L
    from DATA_Analyst_Assistant_Agent.supervisor.insight import InsightGenerator
    monkeypatch.setenv("INSIGHT_CHART_DIR", str(tmp_path))
    monkeypatch.setattr(L, "get_chat_model", lambda *a, **k: FakeLLM(_FINISH_BAD))
    envelope = InsightGenerator().run(_state(), _runtime(FakeAdapter()))
    assert envelope.validation.has_errors              # evidence_present 실패

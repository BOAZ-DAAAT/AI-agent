"""노드 서머리 생성기 테스트 — 전부 FakeLLM/FakeAdapter (실제 LLM·백엔드 호출 0).

검증 대상: 4종 아티팩트(sql/eda/analysis/insight) 증거 추출 / EDA 차트+캡션 수집(key_charts
필드에서, lineage 아님) / 캐시 히트-미스(순서 무시 집합 비교, summary_version) /
findings 섹션 구조(소제목+본문+차트 반복) / 숫자 검증 실패→재시도→폴백(사람이 읽는 라벨) /
등록 아티팩트의 parent_ids·metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from data_agent_backend.models.artifacts import ArtifactRef, ArtifactType

from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.supervisor.summary import generator as gen
from DATA_Analyst_Assistant_Agent.supervisor.summary.evidence import read_node_evidence


# ─────────────────────────────
# 공용 페이크
# ─────────────────────────────
class FakeLLM:
    def __init__(self, *responses: str):
        self._seq = list(responses)
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1
        content = self._seq.pop(0) if self._seq else "{}"
        return SimpleNamespace(content=content)


@dataclass
class _FakeRecord:
    artifact_id: str
    run_id: str
    type: ArtifactType
    metadata: dict
    content: str
    parent_ids: list = field(default_factory=list)
    preview: dict = field(default_factory=dict)

    def ref(self) -> ArtifactRef:
        return ArtifactRef(artifact_id=self.artifact_id, type=self.type)


class FakeAdapter:
    def __init__(self):
        self._store: dict[str, _FakeRecord] = {}
        self._n = 0

    def seed(self, artifact_id: str, run_id: str, artifact_type: ArtifactType, content: str, metadata: dict | None = None) -> None:
        self._store[artifact_id] = _FakeRecord(
            artifact_id=artifact_id, run_id=run_id, type=artifact_type, metadata=metadata or {}, content=content,
        )

    def get_artifact(self, artifact_id: str):
        return self._store[artifact_id]

    def read_artifact_text(self, artifact_id: str) -> str:
        return self._store[artifact_id].content

    def list_artifacts(self, *, run_id: str, artifact_type=None):
        return [
            r for r in self._store.values()
            if r.run_id == run_id and (artifact_type is None or r.type == ArtifactType(artifact_type))
        ]

    def register_artifact(self, run_id, artifact_type, *, content_text=None, content_bytes=None,
                          filename, created_by_tool, parent_ids=None, metadata=None, preview=None, **_kwargs):
        self._n += 1
        record = _FakeRecord(
            artifact_id=f"art-summary-{self._n}", run_id=run_id, type=ArtifactType(artifact_type),
            metadata=dict(metadata or {}), content=content_text or "",
            parent_ids=list(parent_ids or []), preview=dict(preview or {}),
        )
        self._store[record.artifact_id] = record
        return record.ref()


def _runtime(adapter: FakeAdapter) -> AgentRuntime:
    return AgentRuntime(adapter=adapter)


def _valid_llm_json(**overrides) -> str:
    """정상 통과하는 최소 JSON 응답(숫자 없음 — 검증에 안 걸림)."""
    payload = {
        "title": "제목",
        "subtitle": "부제",
        "background": "이 단계를 진행한 배경입니다.",
        "checked_items": ["항목1", "항목2"],
        "findings": [
            {"heading": "소제목", "body": "본문 내용입니다.", "source_label": "요약", "chart_artifact_ids": []},
        ],
        "conclusion": "결론적으로 정리하였습니다.",
        "key_finding": "핵심 발견",
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


_EDA_JSON = json.dumps({
    "final_summary": "toys 카테고리 중심의 매출 집중 구조입니다.",
    "hypotheses": "카테고리 간 매출 편차가 크다.",
    "cautions": [],
    "data_level": {},
    "statistical_metadata": {"row_count": 4},
    "key_charts": [
        {"filename": "chart1.png", "artifact_id": "art-chart-1", "caption": "카테고리별 매출"},
        {"filename": "chart2.png", "artifact_id": "art-chart-2", "caption": "이상치"},
    ],
})
_ANALYSIS_JSON = json.dumps({
    "title": "카테고리별 매출 비교",
    "executive_summary": "toys가 매출 1위입니다.",
    "key_findings": ["toys가 매출 1위(900.0)"],
    "limitations": ["관찰 데이터"],
    "method_notes": [],
    "generated_code": "",
})
_INSIGHT_JSON = json.dumps({
    "answer": "toys가 매출을 주도합니다.",
    "key_insights": ["toys 카테고리가 매출의 60%를 차지"],
    "action_plan": [],
    "limitations": [],
    "charts": [{"artifact_id": "art-insight-chart-1", "filename": "c.png", "title": "카테고리별 비중"}],
})


# ─────────────────────────────
# 증거 추출 — 4종
# ─────────────────────────────
def test_evidence_sql_result_reads_csv_preview():
    adapter = FakeAdapter()
    adapter.seed("s1", "run1", ArtifactType.sql_result, "category,total_sales\ntoys,500.0\nauto,300.0\n",
                {"kind": "sql_result"})
    ev = read_node_evidence(["s1"], _runtime(adapter))
    assert ev.source_kind == "sql_result"
    assert ev.code_used == ""                          # CSV엔 SQL 텍스트가 없음
    assert ev.facts["row_count"] == 2


def test_evidence_sql_plan_extracts_generated_sql_from_flat_field():
    """옛 스키마(최상위 generated_sql) 하위호환 — sql_draft가 없을 때 폴백 경로."""
    adapter = FakeAdapter()
    payload = json.dumps({"generated_sql": "SELECT category, SUM(total_sales) FROM orders GROUP BY category",
                          "target_table": None})
    adapter.seed("p1", "run1", ArtifactType.file, payload, {"kind": "sql_plan"})
    ev = read_node_evidence(["p1"], _runtime(adapter))
    assert ev.source_kind == "sql_plan"
    assert "GROUP BY" in ev.code_used


def test_evidence_sql_plan_extracts_generated_sql_from_nested_sql_draft():
    """실제 sql_plan 구조(재발 방지) — SQL은 최상위가 아니라 sql_draft.sql 안에 중첩되어
    있다. 처음에 이걸 몰라서 항상 빈 code_used가 나오던 버그가 있었다."""
    adapter = FakeAdapter()
    payload = json.dumps({
        "plan": {"original_question": "카테고리별 매출은?", "target_metric": "매출 합계", "grain": "category"},
        "mart_design": {"mart_name": "dm_category_sales"},
        "sql_draft": {
            "sql": "SELECT category, SUM(total_sales) FROM orders GROUP BY category",
            "target_table": "analytics.dm_category_sales",
            "business_grain": "category",
            "reasoning": "카테고리 기준으로 집계",
        },
    })
    adapter.seed("p2", "run1", ArtifactType.file, payload, {"kind": "sql_plan"})
    ev = read_node_evidence(["p2"], _runtime(adapter))
    assert ev.source_kind == "sql_plan"
    assert "GROUP BY" in ev.code_used
    assert ev.facts["target_table"] == "analytics.dm_category_sales"
    assert ev.facts["original_question"] == "카테고리별 매출은?"


def test_evidence_eda_summary_collects_charts_with_captions():
    adapter = FakeAdapter()
    adapter.seed("e1", "run1", ArtifactType.data_profile, _EDA_JSON, {"kind": "eda_summary"})
    ev = read_node_evidence(["e1"], _runtime(adapter))
    assert ev.source_kind == "eda_summary"
    assert [c.artifact_id for c in ev.charts] == ["art-chart-1", "art-chart-2"]
    assert ev.charts[0].caption == "카테고리별 매출"
    assert ev.facts["final_summary"].startswith("toys")
    assert ev.facts["statistical_metadata"] == {"row_count": 4}   # Codex 리뷰: 빠졌던 필드


def test_evidence_analysis_result_code_used_is_empty():
    adapter = FakeAdapter()
    adapter.seed("a1", "run1", ArtifactType.file, _ANALYSIS_JSON, {"kind": "analysis_result"})
    ev = read_node_evidence(["a1"], _runtime(adapter))
    assert ev.source_kind == "analysis_result"
    assert ev.code_used == ""                           # generated_code는 등록 시 항상 빈 문자열
    assert ev.facts["executive_summary"] == "toys가 매출 1위입니다."


def test_evidence_insight_payload_collects_charts():
    adapter = FakeAdapter()
    adapter.seed("i1", "run1", ArtifactType.file, _INSIGHT_JSON, {"kind": "insight_payload"})
    ev = read_node_evidence(["i1"], _runtime(adapter))
    assert ev.source_kind == "insight_payload"
    assert [c.artifact_id for c in ev.charts] == ["art-insight-chart-1"]
    assert ev.charts[0].caption == "카테고리별 비중"
    assert ev.facts["answer"].startswith("toys")


# ─────────────────────────────
# 캐시
# ─────────────────────────────
def test_generate_node_summary_cache_hit_skips_llm(monkeypatch):
    adapter = FakeAdapter()
    adapter.seed("e1", "run1", ArtifactType.data_profile, _EDA_JSON, {"kind": "eda_summary"})
    adapter.register_artifact(
        "run1", ArtifactType.file, content_text="{}", filename="node_summary.json",
        created_by_tool="x", parent_ids=["e1"],
        metadata={"kind": "node_summary", "source_artifact_ids": ["e1"], "summary_version": gen._SUMMARY_VERSION},
    )

    def _poison(*_a, **_k):
        raise AssertionError("캐시 히트인데 LLM을 호출함")

    monkeypatch.setattr(gen, "get_chat_model", _poison)
    ref = gen.generate_node_summary(["e1"], _runtime(adapter))
    assert ref.artifact_id == "art-summary-1"           # 새로 안 만들고 기존 캐시 그대로


def test_generate_node_summary_cache_matches_regardless_of_artifact_id_order(monkeypatch):
    adapter = FakeAdapter()
    adapter.seed("a", "run1", ArtifactType.file, "{}", {"kind": "analysis_result"})
    adapter.seed("b", "run1", ArtifactType.file, "{}", {"kind": "analysis_result"})
    adapter.register_artifact(
        "run1", ArtifactType.file, content_text="{}", filename="node_summary.json",
        created_by_tool="x", parent_ids=["a", "b"],
        metadata={"kind": "node_summary", "source_artifact_ids": ["a", "b"], "summary_version": gen._SUMMARY_VERSION},
    )
    monkeypatch.setattr(gen, "get_chat_model", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM 호출됨")))
    ref = gen.generate_node_summary(["b", "a"], _runtime(adapter))   # 순서를 뒤집어서 호출
    assert ref.artifact_id == "art-summary-1"


def test_generate_node_summary_version_mismatch_regenerates(monkeypatch):
    adapter = FakeAdapter()
    adapter.seed("e1", "run1", ArtifactType.data_profile, _EDA_JSON, {"kind": "eda_summary"})
    adapter.register_artifact(
        "run1", ArtifactType.file, content_text="{}", filename="node_summary.json",
        created_by_tool="x", parent_ids=["e1"],
        metadata={"kind": "node_summary", "source_artifact_ids": ["e1"], "summary_version": 0},  # 옛 버전
    )
    llm = FakeLLM(_valid_llm_json())
    monkeypatch.setattr(gen, "get_chat_model", lambda *a, **k: llm)

    ref = gen.generate_node_summary(["e1"], _runtime(adapter))
    assert ref.artifact_id != "art-summary-1"           # 캐시 무시하고 새로 만듦
    assert llm.calls == 1


# ─────────────────────────────
# findings 구조 + 차트 연결
# ─────────────────────────────
def test_generate_node_summary_builds_findings_with_chart_ids(monkeypatch):
    adapter = FakeAdapter()
    adapter.seed("e1", "run1", ArtifactType.data_profile, _EDA_JSON, {"kind": "eda_summary"})
    response = _valid_llm_json(findings=[
        {"heading": "카테고리별 매출 분포", "body": "toys 카테고리 매출이 두드러집니다.",
         "source_label": "분포 차트", "chart_artifact_ids": ["art-chart-1", "art-chart-2"]},
        {"heading": "이상치 후보", "body": "일부 고액 주문이 관찰됩니다.",
         "chart_artifact_ids": ["art-chart-not-real"]},   # 근거에 없는 id — 걸러져야 함
    ])
    llm = FakeLLM(response)
    monkeypatch.setattr(gen, "get_chat_model", lambda *a, **k: llm)

    ref = gen.generate_node_summary(["e1"], _runtime(adapter))
    content = json.loads(adapter._store[ref.artifact_id].content)

    assert len(content["findings"]) == 2
    assert content["findings"][0]["chart_artifact_ids"] == ["art-chart-1", "art-chart-2"]
    assert content["findings"][1]["chart_artifact_ids"] == []   # 존재하지 않는 차트 id는 제거됨
    assert content["fallback_used"] is False


def test_generate_node_summary_rejects_response_missing_findings(monkeypatch):
    adapter = FakeAdapter()
    adapter.seed("e1", "run1", ArtifactType.data_profile, _EDA_JSON, {"kind": "eda_summary"})
    no_findings = json.dumps({"title": "t", "subtitle": "s", "background": "b",
                              "checked_items": [], "findings": [], "conclusion": "c", "key_finding": "k"})
    llm = FakeLLM(no_findings, no_findings)   # 두 번 다 findings 비어서 실패 -> 폴백
    monkeypatch.setattr(gen, "get_chat_model", lambda *a, **k: llm)

    ref = gen.generate_node_summary(["e1"], _runtime(adapter))
    content = json.loads(adapter._store[ref.artifact_id].content)
    assert content["fallback_used"] is True
    assert llm.calls == 2


# ─────────────────────────────
# 숫자 검증 실패 → 재시도 → 폴백
# ─────────────────────────────
def test_generate_node_summary_falls_back_when_numbers_unverifiable(monkeypatch):
    adapter = FakeAdapter()
    adapter.seed("e1", "run1", ArtifactType.data_profile, _EDA_JSON, {"kind": "eda_summary"})
    fabricated = _valid_llm_json(
        background="매출이 9999.9로 증가했습니다.",
        findings=[{"heading": "h", "body": "근거없는 발견 9999.9", "chart_artifact_ids": []}],
    )
    llm = FakeLLM(fabricated, fabricated)               # 재시도해도 계속 지어냄
    monkeypatch.setattr(gen, "get_chat_model", lambda *a, **k: llm)

    ref = gen.generate_node_summary(["e1"], _runtime(adapter))
    content = json.loads(adapter._store[ref.artifact_id].content)

    assert llm.calls == 2                                # 최초 1회 + 재시도 1회
    assert content["fallback_used"] is True
    assert content["title"]                              # fallback에서도 항상 채워짐
    assert content["background"]
    assert content["source_kind"] == "eda_summary"
    assert "9999.9" not in json.dumps(content, ensure_ascii=False)   # 지어낸 숫자는 최종 결과에 없음


def test_fallback_uses_human_readable_labels_not_raw_keys():
    """폴백에서 raw dict key(final_summary 등)를 그대로 노출하지 않는지 확인(Codex 리뷰)."""
    adapter = FakeAdapter()
    adapter.seed("e1", "run1", ArtifactType.data_profile, _EDA_JSON, {"kind": "eda_summary"})
    ev = read_node_evidence(["e1"], _runtime(adapter))
    result = gen._fallback_result(ev)

    headings = [f.heading for f in result.findings]
    assert "final_summary" not in headings
    assert "핵심 요약" in headings                        # final_summary -> 사람이 읽는 라벨
    assert "statistical_metadata" not in headings
    assert "통계 지표" in headings


def test_generate_node_summary_accepts_valid_result_on_first_try(monkeypatch):
    adapter = FakeAdapter()
    adapter.seed("e1", "run1", ArtifactType.data_profile, _EDA_JSON, {"kind": "eda_summary"})
    llm = FakeLLM(_valid_llm_json(title="EDA 검증 결과", key_finding="toys 매출 집중"))
    monkeypatch.setattr(gen, "get_chat_model", lambda *a, **k: llm)

    ref = gen.generate_node_summary(["e1"], _runtime(adapter))
    content = json.loads(adapter._store[ref.artifact_id].content)
    assert llm.calls == 1
    assert content["fallback_used"] is False
    assert content["key_finding"] == "toys 매출 집중"


# ─────────────────────────────
# 등록 아티팩트 메타데이터
# ─────────────────────────────
def test_generate_node_summary_registers_parent_ids_and_metadata(monkeypatch):
    adapter = FakeAdapter()
    adapter.seed("a1", "run1", ArtifactType.file, _ANALYSIS_JSON, {"kind": "analysis_result"})
    llm = FakeLLM(_valid_llm_json())
    monkeypatch.setattr(gen, "get_chat_model", lambda *a, **k: llm)

    ref = gen.generate_node_summary(["a1"], _runtime(adapter))
    record = adapter._store[ref.artifact_id]
    assert record.parent_ids == ["a1"]
    assert record.metadata["source_artifact_ids"] == ["a1"]
    assert record.metadata["summary_version"] == gen._SUMMARY_VERSION
    assert record.metadata["kind"] == "node_summary"

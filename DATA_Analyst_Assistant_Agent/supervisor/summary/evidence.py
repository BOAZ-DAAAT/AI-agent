"""노드 서머리용 증거 — 아티팩트를 읽어 근거 dict로 정규화한다(계산 없음, 발췌만).

EDA/분석/인사이트는 artifact_ids 중 다룰 줄 아는 kind를 가진 항목 하나를 찾아 읽는다
("한 노드 = 아티팩트 하나" 단순화). SQL만 예외로 sql_plan+sql_result를 병합해서 읽는다
(자세한 이유는 read_node_evidence 참고).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.artifact_data import read_json_artifact
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime

_SQL_KINDS = {"sql_result", "sql_plan"}


@dataclass
class ChartRef:
    """차트 하나 — 캡션이 있어야 LLM이 "어떤 차트를 어느 섹션에 붙일지" 판단할 수 있다."""

    artifact_id: str
    caption: str = ""


@dataclass
class NodeEvidence:
    source_kind: str
    facts: dict[str, Any] = field(default_factory=dict)   # LLM 프롬프트/숫자검증 corpus 용
    code_used: str = ""                                    # LLM이 쓰는 게 아니라 근거에서 그대로 발췌
    charts: list[ChartRef] = field(default_factory=list)


_KNOWN_KINDS = _SQL_KINDS | {"eda_summary", "analysis_result", "insight_payload"}


def read_node_evidence(artifact_ids: list[str], runtime: AgentRuntime) -> NodeEvidence:
    """artifact_ids 중 다룰 줄 아는 kind(위 _KNOWN_KINDS)를 가진 항목을 찾아 읽는다.

    단순히 artifact_ids[0]을 쓰면 안 된다 — 예를 들어 insight는
    [final_report, insight_payload, insight_chart...] 순으로 등록되므로(InsightGenerator.run
    참고) 0번 인덱스는 final_report(마크다운 리포트, 이 함수가 처리 못 하는 kind)가 걸려
    근거가 통째로 비어버린다(실사례로 확인됨).

    SQL만 예외적으로 여러 아티팩트를 병합한다 — sql_agent는 보통 sql_result(CSV, 5행
    미리보기)와 sql_plan(실행된 SQL·grain·근거)을 각각 따로 등록하는데, 실제로는 둘 다
    있어야 "무슨 SQL을 실행해서 어떤 마트가 나왔는지"를 온전히 설명할 수 있다.
    """
    if not artifact_ids:
        return NodeEvidence(source_kind="unknown")

    kinds: dict[str, str] = {}
    for candidate_id in artifact_ids:
        record = runtime.adapter.get_artifact(candidate_id)
        kinds[candidate_id] = str(record.metadata.get("kind") or record.type.value)

    sql_ids = [aid for aid in artifact_ids if kinds[aid] in _SQL_KINDS]
    if sql_ids:
        return _read_sql_evidence_merged(sql_ids, kinds, runtime)

    for candidate_id in artifact_ids:
        kind = kinds[candidate_id]
        if kind in _KNOWN_KINDS:
            return _dispatch_single_kind_evidence(candidate_id, kind, runtime)

    # 아는 kind가 하나도 없으면(완전히 새로운 종류) 첫 항목 kind라도 보고한다.
    return NodeEvidence(source_kind=kinds[artifact_ids[0]] or "unknown")


def _dispatch_single_kind_evidence(artifact_id: str, kind: str, runtime: AgentRuntime) -> NodeEvidence:
    if kind == "eda_summary":
        return _read_eda_evidence(artifact_id, runtime)
    if kind == "analysis_result":
        return _read_analysis_evidence(artifact_id, runtime)
    if kind == "insight_payload":
        return _read_insight_evidence(artifact_id, runtime)
    return NodeEvidence(source_kind=kind or "unknown")


def _read_sql_evidence(artifact_id: str, kind: str, runtime: AgentRuntime) -> NodeEvidence:
    if kind == "sql_plan":
        # 실제 sql_plan 아티팩트를 직접 열어서 구조를 확인한 결과, SQL 텍스트는 최상위
        # generated_sql이 아니라 sql_draft.sql 안에 있다(plan/mart_design/sql_draft/validation
        # 이 각각 중첩된 구조). 처음엔 agents/artifact_data.py::generated_sql_from_artifacts와
        # 같은 필드를 읽는다고 가정했는데, 실제 로컬 데이터로 확인해보니 그 가정 자체가
        # 틀렸었다(그 함수도 이 구조에선 빈 문자열을 반환할 것으로 보이나, 이건 agents/sql
        # 인접 공용 유틸이라 범위 밖 — 여기서는 내 코드만 고친다).
        payload = read_json_artifact(runtime, artifact_id)
        sql_draft = payload.get("sql_draft") or {}
        plan = payload.get("plan") or {}
        generated_sql = str(sql_draft.get("sql") or payload.get("generated_sql") or "")
        target_table = sql_draft.get("target_table") or payload.get("target_table")
        facts = {
            "generated_sql": generated_sql,
            "target_table": target_table,
            "original_question": plan.get("original_question", ""),
            "target_metric": plan.get("target_metric", ""),
            "grain": plan.get("grain", ""),
            "business_grain": sql_draft.get("business_grain", ""),
            "reasoning": sql_draft.get("reasoning", ""),
        }
        return NodeEvidence(source_kind=kind, facts=facts, code_used=generated_sql)

    # sql_result는 CSV — 미리보기만 근거로 삼는다(계산 없음). splitlines()로 단순 텍스트
    # 분해하면 셀 안에 개행이 든 값이 있을 때 행 수가 부풀고 미리보기가 깨지므로,
    # agents/artifact_data.py가 쓰는 것과 같은 pandas 파싱으로 통일한다(Codex 리뷰 반영).
    text = runtime.adapter.read_artifact_text(artifact_id)
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception:  # noqa: BLE001 — 파싱 실패해도 빈 미리보기로 폴백(EmptyDataError 포함)
        df = pd.DataFrame()
    preview = df.head(5).where(pd.notna(df.head(5)), None).to_dict(orient="records")
    facts = {"columns": list(df.columns), "row_count": int(len(df)), "preview": preview}
    return NodeEvidence(source_kind=kind, facts=facts, code_used="")


def _read_sql_evidence_merged(sql_ids: list[str], kinds: dict[str, str], runtime: AgentRuntime) -> NodeEvidence:
    """sql_plan(SQL 텍스트·grain·근거)과 sql_result(CSV 5행 미리보기)를 합친다.

    한 아티팩트만으론 "무슨 SQL을 실행해서 어떤 마트가 나왔는지"를 다 설명 못 한다 —
    sql_plan엔 실제 행 데이터가 없고 sql_result엔 SQL 텍스트가 없다.
    """
    plan_evidence: NodeEvidence | None = None
    result_evidence: NodeEvidence | None = None
    for aid in sql_ids:
        kind = kinds[aid]
        if kind == "sql_plan" and plan_evidence is None:
            plan_evidence = _read_sql_evidence(aid, kind, runtime)
        elif kind == "sql_result" and result_evidence is None:
            result_evidence = _read_sql_evidence(aid, kind, runtime)
        if plan_evidence is not None and result_evidence is not None:
            break

    facts: dict[str, Any] = {}
    code_used = ""
    source_kind = "sql_plan"
    if plan_evidence is not None:
        facts.update(plan_evidence.facts)
        code_used = plan_evidence.code_used
        source_kind = plan_evidence.source_kind
    if result_evidence is not None:
        facts.update(result_evidence.facts)
        if plan_evidence is None:
            source_kind = result_evidence.source_kind
    return NodeEvidence(source_kind=source_kind, facts=facts, code_used=code_used)


def _read_eda_evidence(artifact_id: str, runtime: AgentRuntime) -> NodeEvidence:
    payload = read_json_artifact(runtime, artifact_id)
    facts = {
        "final_summary": payload.get("final_summary", ""),
        "hypotheses": payload.get("hypotheses", ""),
        "primary_hypothesis": payload.get("primary_hypothesis", {}),
        "cautions": payload.get("cautions", []),
        "data_level": payload.get("data_level", {}),
        "statistical_metadata": payload.get("statistical_metadata", {}),   # Codex 리뷰: 빠져있었음
    }
    code_used = ""

    # 차트는 eda_summary의 자식이 아니라 형제(둘 다 SQL 결과물 아래 나란히 걸림,
    # agents/eda/agent.py의 register_key_chart_artifacts(..., source_ids, ...) 확인).
    # 그래서 lineage 조회 대신 이미 읽은 JSON 안의 key_charts 필드를 그대로 쓴다.
    # 캡션까지 넘겨야 LLM이 어느 섹션에 어떤 차트를 붙일지 판단할 수 있다(Codex 리뷰).
    key_charts = payload.get("key_charts") or []
    charts = [
        ChartRef(artifact_id=str(c["artifact_id"]), caption=str(c.get("caption") or ""))
        for c in key_charts if isinstance(c, dict) and c.get("artifact_id")
    ]

    return NodeEvidence(source_kind="eda_summary", facts=facts, code_used=code_used, charts=charts)


def _read_analysis_evidence(artifact_id: str, runtime: AgentRuntime) -> NodeEvidence:
    payload = read_json_artifact(runtime, artifact_id)
    # 실제 필드명 확인(agents/analysis/agent.py:_public_result_payload) — method_summary가
    # 아니라 title/executive_summary/method_notes다. generated_code는 등록 시 항상 ""로
    # 비워지므로(같은 파일 확인) code_used는 자연스럽게 빈 문자열이 된다.
    facts = {
        "title": payload.get("title", ""),
        "executive_summary": payload.get("executive_summary", ""),
        "key_findings": payload.get("key_findings", []),
        "limitations": payload.get("limitations", []),
        "method_notes": payload.get("method_notes", []),
        "method_decision": payload.get("method_decision") or {},   # selected_method/rationale/... (근거 그대로)
        "hypothesis_tests": payload.get("hypothesis_tests", []),
        "evidence_tables": _analysis_evidence_tables(payload.get("evidence_tables")),
        "interpretation": payload.get("interpretation", []),
    }
    code_used = str(payload.get("generated_code") or "")
    charts: list[ChartRef] = []
    seen_chart_ids: set[str] = set()
    for visual in payload.get("visual_evidence") or []:
        if not isinstance(visual, dict) or visual.get("status") != "read_success":
            continue
        chart_id = str(visual.get("chart_artifact_id") or "").strip()
        if not chart_id or chart_id in seen_chart_ids:
            continue
        caption = str(
            visual.get("multimodal_summary")
            or visual.get("title")
            or visual.get("filename")
            or chart_id
        ).strip()
        charts.append(ChartRef(artifact_id=chart_id, caption=caption))
        seen_chart_ids.add(chart_id)
    return NodeEvidence(source_kind="analysis_result", facts=facts, code_used=code_used, charts=charts)


def _analysis_evidence_tables(raw: Any) -> list[dict[str, Any]]:
    """분석 결과표를 화면 근거로 쓸 수 있는 최대 5행 미리보기로 제한한다."""
    if not isinstance(raw, list):
        return []
    tables: list[dict[str, Any]] = []
    for index, table in enumerate(raw):
        if not isinstance(table, dict):
            continue
        rows = [row for row in (table.get("rows") or []) if isinstance(row, dict)][:5]
        columns = [str(column) for column in (table.get("columns") or [])]
        if not columns and rows:
            columns = list(rows[0].keys())
        tables.append({
            "title": str(table.get("title") or f"근거표 {index + 1}"),
            "columns": columns,
            "rows": rows,
        })
    return tables


def _read_insight_evidence(artifact_id: str, runtime: AgentRuntime) -> NodeEvidence:
    payload = read_json_artifact(runtime, artifact_id)
    labels = payload.get("evidence_labels") or {}
    sources = payload.get("evidence_sources") or []
    facts = {
        "answer": payload.get("answer", ""),
        "key_insights": payload.get("key_insights", []),
        "action_plan": payload.get("action_plan", []),
        "limitations": payload.get("limitations", []),
        # 사람이 읽는 라벨로 변환해서 넘긴다(원본은 artifact_id라 LLM/화면에 그대로 노출하면 안 읽힘).
        "evidence_labels": [str(labels.get(a, a)) for a in sources] if sources else list(labels.values()),
    }
    # insight ChartEntry는 caption이 아니라 title 필드에 설명이 있다(supervisor/insight/schemas.py).
    charts_raw = payload.get("charts") or []
    charts = [
        ChartRef(artifact_id=str(c["artifact_id"]), caption=str(c.get("title") or ""))
        for c in charts_raw if isinstance(c, dict) and c.get("artifact_id")
    ]
    return NodeEvidence(source_kind="insight_payload", facts=facts, code_used="", charts=charts)

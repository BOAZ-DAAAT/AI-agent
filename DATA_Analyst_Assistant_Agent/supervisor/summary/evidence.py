"""노드 서머리용 증거 — 아티팩트를 읽어 근거 dict로 정규화한다(계산 없음, 발췌만).

이번 단계는 artifact_ids 리스트의 첫 항목만 실질적으로 다룬다("한 노드 = 아티팩트 하나"
단순화, evidence.py:read_node_evidence 시그니처는 나중에 "노드 하나가 여러 아티팩트를
대표"하는 경우로 확장하기 위해 처음부터 리스트로 받는다 — 다중 병합 로직은 지금 안 만듦).
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


def read_node_evidence(artifact_ids: list[str], runtime: AgentRuntime) -> NodeEvidence:
    if not artifact_ids:
        return NodeEvidence(source_kind="unknown")

    artifact_id = artifact_ids[0]
    record = runtime.adapter.get_artifact(artifact_id)
    kind = str(record.metadata.get("kind") or record.type.value)

    if kind in _SQL_KINDS:
        return _read_sql_evidence(artifact_id, kind, runtime)
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


def _read_eda_evidence(artifact_id: str, runtime: AgentRuntime) -> NodeEvidence:
    payload = read_json_artifact(runtime, artifact_id)
    facts = {
        "final_summary": payload.get("final_summary", ""),
        "hypotheses": payload.get("hypotheses", ""),
        "cautions": payload.get("cautions", []),
        "data_level": payload.get("data_level", {}),
        "statistical_metadata": payload.get("statistical_metadata", {}),   # Codex 리뷰: 빠져있었음
    }
    # adhoc_analysis에 코드가 있으면 발췌(없으면 빈 문자열 — 억지로 만들지 않음)
    adhoc = (payload.get("statistical_metadata") or {}).get("adhoc_analysis") or {}
    code_used = str(adhoc.get("code") or "")

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
    }
    code_used = str(payload.get("generated_code") or "")
    return NodeEvidence(source_kind="analysis_result", facts=facts, code_used=code_used)


def _read_insight_evidence(artifact_id: str, runtime: AgentRuntime) -> NodeEvidence:
    payload = read_json_artifact(runtime, artifact_id)
    facts = {
        "answer": payload.get("answer", ""),
        "key_insights": payload.get("key_insights", []),
        "action_plan": payload.get("action_plan", []),
        "limitations": payload.get("limitations", []),
    }
    # insight ChartEntry는 caption이 아니라 title 필드에 설명이 있다(supervisor/insight/schemas.py).
    charts_raw = payload.get("charts") or []
    charts = [
        ChartRef(artifact_id=str(c["artifact_id"]), caption=str(c.get("title") or ""))
        for c in charts_raw if isinstance(c, dict) and c.get("artifact_id")
    ]
    return NodeEvidence(source_kind="insight_payload", facts=facts, code_used="", charts=charts)

"""경로 전체 근거 — SQL→EDA→분석→인사이트 중 실제로 존재하는 아티팩트를 단계별로 정규화한다.

summary/evidence.py와 같은 "아티팩트를 읽기만 한다(계산 없음)" 원칙을 따르되, 노드 하나가
아니라 경로 전체를 한 번에 모은다. report/insight/summary 독립성 원칙에 따라 summary의
코드를 import하지 않고 원본 아티팩트를 직접 다시 읽는다(서머리를 거쳐 정보가 압축되는 걸
막기 위함 — summary는 UI 패널 하나에 맞추려고 일부러 줄여 쓰지만, 리포트는 원본 그대로
LLM에게 준다).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.artifact_data import read_json_artifact
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime

_STAGE_ORDER = ("sql", "eda", "analysis", "insight")
_KIND_TO_STAGE = {
    "sql_result": "sql",
    "sql_plan": "sql",
    "eda_summary": "eda",
    "analysis_result": "analysis",
    "insight_payload": "insight",
}
_STAGE_LABELS = {
    "sql": "SQL 조회",
    "eda": "EDA 검증",
    "analysis": "분석",
    "insight": "인사이트",
}


@dataclass
class ChartRef:
    """차트 하나 — 캡션 + 어느 단계 것인지가 있어야 LLM이 어느 섹션에 붙일지 판단할 수 있다."""

    artifact_id: str
    caption: str = ""
    stage: str = ""


@dataclass
class StageEvidence:
    stage: str                                          # sql | eda | analysis | insight
    label: str
    facts: dict[str, Any] = field(default_factory=dict)
    code_used: str = ""
    charts: list[ChartRef] = field(default_factory=list)


@dataclass
class PathEvidence:
    user_question: str = ""
    stages: list[StageEvidence] = field(default_factory=list)   # 존재하는 단계만, sql→eda→analysis→insight 순서 유지

    @property
    def charts(self) -> list[ChartRef]:
        return [c for stage in self.stages for c in stage.charts]

    @property
    def present_stage_names(self) -> list[str]:
        return [s.stage for s in self.stages]


def read_path_evidence(artifact_ids: list[str], runtime: AgentRuntime) -> PathEvidence:
    """artifact_ids(순서 무관, 경로 위 아티팩트 전부)를 종류별로 묶어 단계 순서대로 정리한다.

    지금은 분기가 없어 단계당 인사이트/분석/EDA 아티팩트가 보통 하나뿐이지만, 나중에
    분기 트리가 생겨도 이 함수의 계약(artifact_ids 리스트를 받아 PathEvidence를 만든다)은
    그대로 유지된다 — 달라지는 건 "어떤 artifact_ids를 넘기느냐"(지금은 이 런의 전부,
    나중엔 리프에서 부모를 거슬러 올라간 결과)뿐이다.
    """
    by_stage: dict[str, list[tuple[str, str]]] = {stage: [] for stage in _STAGE_ORDER}
    for artifact_id in artifact_ids:
        if not artifact_id:
            continue
        try:
            record = runtime.adapter.get_artifact(artifact_id)
        except Exception:  # noqa: BLE001 — 조회 실패한 근거는 그냥 건너뛴다(전체를 막지 않음)
            continue
        kind = str(record.metadata.get("kind") or record.type.value)
        stage = _KIND_TO_STAGE.get(kind)
        if stage is None:
            continue
        by_stage[stage].append((artifact_id, kind))

    user_question = ""
    stages: list[StageEvidence] = []
    for stage in _STAGE_ORDER:
        entries = by_stage[stage]
        if not entries:
            continue
        if stage == "sql":
            evidence, question = _read_sql_stage(entries, runtime)
            user_question = user_question or question
        elif stage == "eda":
            evidence = _read_eda_stage(entries, runtime)
        elif stage == "analysis":
            evidence = _read_analysis_stage(entries, runtime)
        else:
            evidence = _read_insight_stage(entries, runtime)
        stages.append(evidence)

    return PathEvidence(user_question=user_question, stages=stages)


def _read_sql_stage(entries: list[tuple[str, str]], runtime: AgentRuntime) -> tuple[StageEvidence, str]:
    plan_ids = [aid for aid, kind in entries if kind == "sql_plan"]
    result_ids = [aid for aid, kind in entries if kind == "sql_result"]

    question = ""
    facts: dict[str, Any] = {}
    code_used = ""
    # sql_plan이 여러 개면(재시도 등) 마지막 걸 최종본으로 본다.
    for artifact_id in plan_ids:
        payload = read_json_artifact(runtime, artifact_id)
        sql_draft = payload.get("sql_draft") or {}
        plan = payload.get("plan") or {}
        generated_sql = str(sql_draft.get("sql") or payload.get("generated_sql") or "")
        question = str(plan.get("original_question") or "") or question
        facts = {
            "generated_sql": generated_sql,
            "target_table": sql_draft.get("target_table") or payload.get("target_table"),
            "target_metric": plan.get("target_metric", ""),
            "grain": plan.get("grain", ""),
            "business_grain": sql_draft.get("business_grain", ""),
            "reasoning": sql_draft.get("reasoning", ""),
        }
        code_used = generated_sql or code_used

    # sql_result가 여러 개면(마트 경로 등) 가장 실질적인(행 많은) 프레임을 미리보기로 쓴다
    # (insight/evidence.py의 _best_dataframe과 같은 이유 — 상태 메시지성 1행 CSV를 피함).
    best_preview: dict[str, Any] = {}
    best_row_count = -1
    for artifact_id in result_ids:
        try:
            text = runtime.adapter.read_artifact_text(artifact_id)
            df = pd.read_csv(io.StringIO(text))
        except Exception:  # noqa: BLE001
            continue
        if len(df) > best_row_count:
            best_row_count = len(df)
            best_preview = {
                "columns": list(df.columns),
                "row_count": int(len(df)),
                "preview": df.head(5).where(pd.notna(df.head(5)), None).to_dict(orient="records"),
            }
    if best_preview:
        facts["result_preview"] = best_preview

    return StageEvidence(stage="sql", label=_STAGE_LABELS["sql"], facts=facts, code_used=code_used), question


def _read_eda_stage(entries: list[tuple[str, str]], runtime: AgentRuntime) -> StageEvidence:
    artifact_id = entries[0][0]
    payload = read_json_artifact(runtime, artifact_id)
    facts = {
        "final_summary": payload.get("final_summary", ""),
        "hypotheses": payload.get("hypotheses", ""),
        "cautions": payload.get("cautions", []),
        "data_level": payload.get("data_level", {}),
        "statistical_metadata": payload.get("statistical_metadata", {}),
    }
    adhoc = (payload.get("statistical_metadata") or {}).get("adhoc_analysis") or {}
    code_used = str(adhoc.get("code") or "")

    key_charts = payload.get("key_charts") or []
    charts = [
        ChartRef(artifact_id=str(c["artifact_id"]), caption=str(c.get("caption") or ""), stage="eda")
        for c in key_charts if isinstance(c, dict) and c.get("artifact_id")
    ]
    return StageEvidence(stage="eda", label=_STAGE_LABELS["eda"], facts=facts, code_used=code_used, charts=charts)


def _read_analysis_stage(entries: list[tuple[str, str]], runtime: AgentRuntime) -> StageEvidence:
    artifact_id = entries[0][0]
    payload = read_json_artifact(runtime, artifact_id)
    facts = {
        "title": payload.get("title", ""),
        "executive_summary": payload.get("executive_summary", ""),
        "key_findings": payload.get("key_findings", []),
        "limitations": payload.get("limitations", []),
        "method_notes": payload.get("method_notes", []),
    }
    # generated_code는 등록 시 항상 ""로 비워진다(summary/evidence.py와 동일 확인 사항).
    code_used = str(payload.get("generated_code") or "")
    return StageEvidence(stage="analysis", label=_STAGE_LABELS["analysis"], facts=facts, code_used=code_used)


def _read_insight_stage(entries: list[tuple[str, str]], runtime: AgentRuntime) -> StageEvidence:
    artifact_id = entries[0][0]
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
        ChartRef(artifact_id=str(c["artifact_id"]), caption=str(c.get("title") or ""), stage="insight")
        for c in charts_raw if isinstance(c, dict) and c.get("artifact_id")
    ]
    return StageEvidence(stage="insight", label=_STAGE_LABELS["insight"], facts=facts, code_used="", charts=charts)

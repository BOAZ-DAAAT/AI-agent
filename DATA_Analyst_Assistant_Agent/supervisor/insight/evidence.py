"""증거팩 — 인사이트가 인용할 수 있는 '검증된 재료'만 모은다.

상류(SQL/EDA/Analysis)가 백엔드에 남긴 아티팩트를 **읽기만** 한다(다른 에이전트 코드 수정 없음).
LLM 답변의 모든 숫자는 이 팩(+게이트 통과 compute 결과)에서만 나와야 하고,
그 검증은 verify.py 가 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.artifact_data import (
    generated_sql_from_artifacts,
    read_json_artifact,
    read_sql_result_csvs,
)
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState


@dataclass
class EvidencePack:
    user_question: str
    route_kind: str                                   # simple | comprehensive (참고 힌트 — 분기 룰 아님)
    generated_sql: str = ""
    df: pd.DataFrame | None = None                    # SQL 결과 원본 (look/compute/chart 대상)
    table_summary: dict[str, Any] = field(default_factory=dict)
    eda: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    eda_raw: dict[str, Any] = field(default_factory=dict)
    analysis_raw: dict[str, Any] = field(default_factory=dict)
    analysis_debug: dict[str, Any] = field(default_factory=dict)
    raw_artifact_ids: dict[str, str] = field(default_factory=dict)
    source_artifact_ids: list[str] = field(default_factory=list)
    source_labels: dict[str, str] = field(default_factory=dict)   # artifact_id → 사람이 읽는 라벨


def build_evidence_pack(state: OrchestrationState, runtime: AgentRuntime) -> EvidencePack:
    """상류 아티팩트를 읽어 증거팩을 조립한다. 없는 재료는 빈 채로 둔다(루프가 알아서 판단)."""
    csvs = read_sql_result_csvs(state, runtime)
    df = _best_dataframe(csvs)
    source_ids = [c.artifact_id for c in csvs if c.error is None]
    labels = {c.artifact_id: "SQL 결과 테이블" for c in csvs if c.error is None}

    eda_id, eda = _read_first_json_artifact(state, runtime, "eda_agent", source_ids)
    analysis_id, analysis = _read_first_json_artifact(state, runtime, "analysis_agent", source_ids)
    analysis_debug_id, analysis_debug = _read_analysis_debug_artifact(state, runtime, analysis, source_ids)
    for aid in source_ids:                             # 사람이 읽는 출처 라벨 (리포트용)
        labels.setdefault(aid, "EDA 분석 요약" if aid in (state.artifact_ids.get("eda_agent") or [])
                          else "분석 결과" if aid in (state.artifact_ids.get("analysis_agent") or [])
                          else "상류 아티팩트")

    return EvidencePack(
        user_question=state.user_query,
        route_kind=state.route_kind or "simple",
        generated_sql=generated_sql_from_artifacts(state, runtime),
        df=df if not df.empty else None,
        table_summary=_summarize_table(df),
        eda=_slim_eda(eda),
        analysis=_slim_analysis(analysis),
        eda_raw=eda,
        analysis_raw=analysis,
        analysis_debug=analysis_debug,
        raw_artifact_ids={
            key: value for key, value in {
                "eda": eda_id,
                "analysis": analysis_id,
                "analysis_debug": analysis_debug_id,
            }.items() if value
        },
        source_artifact_ids=source_ids,
        source_labels=labels,
    )


def _best_dataframe(csvs) -> pd.DataFrame:
    """sql_result 가 여러 개일 때 가장 실질적인(행 많은) 프레임을 고른다.

    mart 경로에서는 '마트 생성 완료' 1행짜리 상태 메시지 CSV 가 먼저 오고 실제 데이터가
    뒤에 오는데, '첫 번째' 선택은 상태 메시지를 집어 분석 불능이 된다 (E2E 에서 실측).
    """
    frames = [c.dataframe for c in csvs if c.error is None and not c.dataframe.empty]
    if not frames:
        return pd.DataFrame()
    return max(frames, key=len)


def _read_first_json_artifact(state: OrchestrationState, runtime: AgentRuntime,
                              agent_name: str, source_ids: list[str]) -> tuple[str, dict[str, Any]]:
    """Return the first readable JSON payload for an agent plus its artifact id."""
    for artifact_id in state.artifact_ids.get(agent_name, []) or []:
        try:
            payload = read_json_artifact(runtime, artifact_id)
        except Exception:  # noqa: BLE001
            continue
        if payload:
            source_ids.append(artifact_id)
            return artifact_id, payload
    return "", {}


def _read_first_json(state: OrchestrationState, runtime: AgentRuntime,
                     agent_name: str, source_ids: list[str]) -> dict[str, Any]:
    """해당 에이전트의 첫 JSON 아티팩트를 읽는다. 읽기 실패는 빈 dict(파이프라인 안 막음)."""
    for artifact_id in state.artifact_ids.get(agent_name, []) or []:
        try:
            payload = read_json_artifact(runtime, artifact_id)
        except Exception:  # noqa: BLE001
            continue
        if payload:
            source_ids.append(artifact_id)
            return payload
    return {}


def _read_analysis_debug_artifact(
    state: OrchestrationState,
    runtime: AgentRuntime,
    analysis_payload: dict[str, Any],
    source_ids: list[str],
) -> tuple[str, dict[str, Any]]:
    """Read analysis_debug so insight can inspect raw code/statistics on demand."""
    candidate_ids: list[str] = []
    debug_id = str(analysis_payload.get("debug_artifact_id") or "").strip()
    if debug_id:
        candidate_ids.append(debug_id)
    candidate_ids.extend(str(aid) for aid in state.artifact_ids.get("analysis_agent", []) or [])

    seen: set[str] = set()
    for artifact_id in candidate_ids:
        if not artifact_id or artifact_id in seen:
            continue
        seen.add(artifact_id)
        try:
            payload = read_json_artifact(runtime, artifact_id)
        except Exception:  # noqa: BLE001
            continue
        if not payload:
            continue
        try:
            record = runtime.adapter.get_artifact(artifact_id)
            kind = str(getattr(record, "metadata", {}).get("kind") or "")
        except Exception:  # noqa: BLE001
            kind = ""
        if artifact_id == debug_id or kind == "analysis_debug" or "raw_statistics" in payload:
            if artifact_id not in source_ids:
                source_ids.append(artifact_id)
            return artifact_id, payload
    return "", {}


def _summarize_table(df: pd.DataFrame) -> dict[str, Any]:
    """LLM 첫 관찰용 테이블 요약. head/describe 값은 증거 corpus 에도 그대로 들어간다."""
    if df is None or df.empty:
        return {}
    numeric = df.select_dtypes(include="number")
    describe = {}
    if not numeric.empty:
        describe = {
            col: {k: (None if pd.isna(v) else round(float(v), 4))
                  for k, v in stats.items()
                  if k in {"count", "mean", "std", "min", "50%", "max"}}
            for col, stats in numeric.describe().to_dict().items()
        }
    return {
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "numeric_describe": describe,
        "head": df.head(5).where(pd.notna(df.head(5)), None).to_dict(orient="records"),
    }


def _slim_eda(payload: dict[str, Any]) -> dict[str, Any]:
    """eda_summary 에서 인사이트에 쓸 부분만. statistical_metadata 는 look 도구·검증 corpus 용으로 보존."""
    if not payload:
        return {}
    return {
        "final_summary": payload.get("final_summary", ""),
        "hypotheses": payload.get("hypotheses", ""),
        "cautions": payload.get("cautions", []),
        "data_level": payload.get("data_level", {}),
        "statistical_metadata": payload.get("statistical_metadata", {}),
        "key_charts": payload.get("key_charts", []),
    }


def _slim_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    """분석 result 에서 인사이트에 쓸 부분만 (텍스트 findings — 분석은 결론차트를 안 그림)."""
    if not payload:
        return {}
    return {
        "method_summary": payload.get("method_summary", ""),
        "key_findings": payload.get("key_findings", []),
        "limitations": payload.get("limitations", []),
        "data_quality_notes": payload.get("data_quality_notes", []),
    }

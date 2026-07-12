from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from data_agent_backend.models.artifacts import ArtifactRef, ArtifactType

from DATA_Analyst_Assistant_Agent.agents.artifact_data import CsvArtifactData, load_analysis_inputs
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.agents.eda._runtime import EdaContext, reset_context, set_context
from DATA_Analyst_Assistant_Agent.shared.contracts import (
    AgentEnvelope,
    LocalCheck,
    OrchestrationState,
    RetryHint,
    ValidationBlock,
)


def register_key_chart_artifacts(runtime, state, chart_paths, parent_ids, context, captions=None):
    """key 차트 PNG를 아티팩트로 등록한다 — **이상적 형태(content_bytes)** 로 호출.

    adapter가 아직 바이너리(content_bytes)를 지원하지 않으면 가드로 잡아 artifact_id=None 폴백한다
    (파이프라인 안 막음). 백엔드가 adapter에 content_bytes를 열면 코드 변경 없이 실제 등록이 작동한다.
    분석 에이전트는 경로 대신 artifact_id로 차트를 로드(멀티모달)한다.
    반환: (entries, refs) — entries=[{"filename","artifact_id","caption"}, ...](payload용),
    refs=등록 성공한 ArtifactRef 목록(AgentEnvelope.artifact_refs용 — 빠지면 state.artifact_ids에
    안 잡혀서 차트가 "만들어졌지만 추적 안 되는" 상태가 된다, #120).
    """
    captions = captions or {}
    entries: list[dict[str, Any]] = []
    refs: list[ArtifactRef] = []
    for path in chart_paths or []:
        filename = os.path.basename(path)
        caption = captions.get(filename, "")
        artifact_id = None
        try:
            with open(path, "rb") as fh:
                png_bytes = fh.read()
            ref = runtime.adapter.register_artifact(
                state.run_id,
                ArtifactType.chart,
                content_bytes=png_bytes,   # 이상형 — 백엔드가 content_bytes 열면 실제 저장
                filename=filename,
                created_by_tool="DATA_Analyst_Assistant_Agent.eda.lang_graph",
                context=context,
                parent_ids=parent_ids,
                metadata={"caption": caption},   # 선정 이유 — 분석 멀티모달 읽기의 설명서(#71 B)
            )
            artifact_id = ref.artifact_id
            refs.append(ref)
        except Exception:  # noqa: BLE001  # adapter 미지원/파일 없음 등 → 폴백
            artifact_id = None
        entries.append({"filename": filename, "artifact_id": artifact_id, "caption": caption})
    return entries, refs


class EDAAgent:
    name = "eda_agent"

    def run(self, state: OrchestrationState, runtime: AgentRuntime) -> AgentEnvelope:
        context = runtime.context(state, node_name=self.name, tool_name="eda_agent.lang_graph")
        # comprehensive(마트) 경로면 analytics 스키마에서 마트를 DB 로 직접 조회하고,
        # simple 경로/조회 실패 시 sql_result CSV 아티팩트로 폴백한다(반환형은 동일).
        csvs = load_analysis_inputs(state, runtime)
        source_ids = [csv.artifact_id for csv in csvs if csv.artifact_id]
        profile = profile_from_csv_artifacts(csvs)

        # 원본 LangGraph EDA 실행 (planner → 분석 노드 → insight/hypothesis → chart_selector)
        eda_result = self._run_eda_graph(csvs, state)

        # key 차트 PNG를 아티팩트로 등록(이상형+가드) → 경로 대신 {filename, artifact_id}로 전달
        key_chart_entries, key_chart_refs = register_key_chart_artifacts(
            runtime, state, eda_result.get("key_charts", []), source_ids, context,
            captions=eda_result.get("key_chart_captions", {}))

        # codegen 탈출구가 도메인 밖으로 판정하면 top-level 플래그로 정직하게 노출한다
        # (성공 결과는 statistical_metadata.adhoc_analysis에 편입됨). 분석 에이전트가 라우팅에 씀.
        codegen = eda_result.get("codegen", {}) or {}
        out_of_domain = (
            {"reason": codegen.get("reason", ""), "user_question": codegen.get("user_question", "")}
            if codegen.get("status") == "out_of_domain" else None
        )

        payload = {
            "run_id": state.run_id,
            "source_artifacts": source_ids,
            "profile": profile,
            "user_question": eda_result.get("user_question", state.user_query),
            "analysis_plan": eda_result.get("analysis_plan", {}),
            "insight_result": eda_result.get("insight_result", ""),
            "hypotheses": eda_result.get("hypotheses", ""),
            "final_summary": eda_result.get("final_summary", ""),
            "analysis_target": eda_result.get("analysis_target", ""),
            "data_level": eda_result.get("data_level", {}),
            "cautions": eda_result.get("cautions", []),
            "analysis_constraints": eda_result.get("analysis_constraints", []),
            "statistical_metadata": eda_result.get("statistical_metadata", {}),
            "key_charts": key_chart_entries,
            "out_of_domain": out_of_domain,
            "error_log": eda_result.get("error_log", []),
        }
        ref = runtime.adapter.register_artifact(
            state.run_id,
            ArtifactType.data_profile,
            content_text=json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            filename="eda_summary.json",
            created_by_tool="DATA_Analyst_Assistant_Agent.eda.lang_graph",
            context=context,
            parent_ids=source_ids,
            metadata={"kind": "eda_summary", "source_artifact_count": len(source_ids)},
            preview={
                "row_count": profile["row_count"],
                "columns": profile["columns"],
                "final_summary": payload["final_summary"],
                "key_charts": payload["key_charts"],
                "quality_status": profile["quality_status"],
            },
        )
        return AgentEnvelope(
            agent_name=self.name,
            summary=payload["final_summary"] or "EDA LangGraph analysis completed.",
            artifact_refs=[ref, *key_chart_refs],
            validation=ValidationBlock(
                local_checks=run_eda_self_check(source_ids, profile, payload["cautions"])
            ),
            retry_hint=_build_retry_hint(eda_result.get("validation_result", {})),
            # 서브에이전트는 핸드오프를 갖지 않는다 — 다음 단계 라우팅은 메인(supervisor)의 몫.
        )

    def _run_eda_graph(self, csvs: list[Any], state: OrchestrationState) -> dict[str, Any]:
        """CSV 아티팩트로 DataFrame을 만들고 원본 EDA LangGraph를 실행한다."""
        from DATA_Analyst_Assistant_Agent.agents.eda.graph import build_app

        frames = [csv.dataframe for csv in csvs if csv.error is None and not csv.dataframe.empty]
        if not frames:
            raise RuntimeError("No non-empty SQL CSV artifact was available for EDA.")
        # mart 경로에선 '마트 생성 완료' 상태 메시지 CSV(1행)가 섞여 온다 — concat 하면
        # col_1 쓰레기차트·data_level 오판·key_col 오염으로 comparison 이 전멸한다(E2E 실측).
        # 스키마가 같은 프레임만 합치고, 아니면 실질 데이터(최대 행) 프레임을 쓴다.
        main = max(frames, key=len)
        same_schema = [f for f in frames if list(f.columns) == list(main.columns)]
        df = pd.concat(same_schema, ignore_index=True) if len(same_schema) > 1 else main

        # 원본 모듈 전역(_df 등)을 대체하는 실행 컨텍스트. df 만 채우고
        # key/measure/time 컬럼은 load_mart 노드가 확정한다.
        set_context(EdaContext(df=df, question_type=state.route_kind or ""))
        # 앞단이 넘긴 의미 힌트를 EDA로 전달(있으면 줍고 없으면 폴백). 앞단(오케스트레이터)은
        # "매출"/"월"/None 수준이라 대개 비어 옴 → 가설 노드가 priority_metrics로 폴백한다.
        plan = state.plan
        plan_metric = (plan.metric if plan and plan.metric else "") or ""
        plan_dimension = (plan.dimension if plan and plan.dimension else "") or ""
        # GE 정합성 스코핑용 원천 테이블 + grain 교차검증용 선언 grain(있으면 줍고 없으면 폴백).
        plan_source_tables = list(plan.source_tables) if plan and plan.source_tables else []
        plan_business_grain = (plan.business_grain if plan and plan.business_grain else "") or ""
        try:
            app = build_app()
            result = app.invoke(
                {
                    "user_question": state.user_query,
                    "target_table": "",
                    "mart_design": {},
                    "question_type": state.route_kind or "",
                    "plan_metric": plan_metric,
                    "plan_dimension": plan_dimension,
                    "plan_source_tables": plan_source_tables,
                    "plan_business_grain": plan_business_grain,
                    "error_log": [],
                }
            )
        finally:
            reset_context()
        return result


# ─────────────────────────────
# 백엔드 핸드오프 헬퍼 (구 profiler.py / self_check.py 통합)
# ─────────────────────────────
def profile_from_csv_artifacts(csvs: list[CsvArtifactData]) -> dict[str, Any]:
    key_issues: list[str] = []
    source_artifacts = [csv.artifact_id for csv in csvs]
    read_errors = {csv.artifact_id: csv.error for csv in csvs if csv.error}

    if not csvs:
        key_issues.append("No SQL result artifact was available for EDA.")
        df = pd.DataFrame()
    else:
        frames = [csv.dataframe for csv in csvs if csv.error is None]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if read_errors:
        key_issues.extend(f"{artifact_id}: {error}" for artifact_id, error in read_errors.items())
    if csvs and len(df.columns) == 0:
        key_issues.append("SQL result CSV did not include columns.")
    if csvs and len(df) == 0:
        key_issues.append("SQL result CSV contains zero data rows.")

    null_counts = {column: int(count) for column, count in df.isna().sum().items()} if len(df.columns) else {}
    unique_counts = {column: int(df[column].nunique(dropna=True)) for column in df.columns}
    numeric_summary = _numeric_summary(df)
    categorical_top_values = _categorical_top_values(df)

    if not csvs:
        quality_status = "unavailable"
        recommended_next_steps = ["run_sql_preview", "clarify_data_source"]
    elif key_issues:
        quality_status = "needs_review"
        recommended_next_steps = ["review_csv_artifact", "rerun_or_refine_sql"]
    else:
        quality_status = "usable"
        recommended_next_steps = ["continue_to_analysis", "document_limitations"]

    return {
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "null_counts": null_counts,
        "unique_counts": unique_counts,
        "numeric_summary": numeric_summary,
        "categorical_top_values": categorical_top_values,
        "sample_available": len(df) > 0,
        "quality_status": quality_status,
        "key_issues": key_issues,
        "recommended_next_steps": recommended_next_steps,
        "source_artifacts": source_artifacts,
    }


def _numeric_summary(df: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return {}
    summary = numeric_df.describe().to_dict()
    return {
        column: {
            stat: (None if pd.isna(value) else float(value))
            for stat, value in stats.items()
            if stat in {"count", "mean", "std", "min", "25%", "50%", "75%", "max"}
        }
        for column, stats in summary.items()
    }


def _categorical_top_values(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    categorical: dict[str, dict[str, int]] = {}
    for column in df.select_dtypes(exclude="number").columns:
        counts = df[column].fillna("<NA>").astype(str).value_counts(dropna=False).head(5)
        categorical[column] = {str(value): int(count) for value, count in counts.items()}
    return categorical


def run_eda_self_check(
    source_artifact_ids: list[str], profile: dict, cautions: list[dict] | None = None
) -> list[LocalCheck]:
    cautions = cautions or []
    validator_failure = next(
        (
            c.get("message_ko", "")
            for c in cautions
            if isinstance(c, dict) and c.get("code") == "EDA_SELF_VALIDATION_FAILED"
        ),
        None,
    )
    return [
        LocalCheck(
            name="source_artifact_present",
            passed=bool(source_artifact_ids),
            severity="error" if not source_artifact_ids else "info",
            detail="EDA requires at least one upstream SQL artifact.",
        ),
        LocalCheck(
            name="profile_generated",
            passed=bool(profile),
            severity="error" if not profile else "info",
            detail="EDA profile payload was generated.",
        ),
        LocalCheck(
            name="columns_present",
            passed=bool(profile.get("columns")),
            severity="warning" if not profile.get("columns") else "info",
            detail="Column metadata should be available for downstream analysis.",
        ),
        LocalCheck(
            name="eda_self_validation",
            passed=validator_failure is None,
            severity="error" if validator_failure is not None else "info",
            detail=validator_failure or "EDA internal validator passed.",
        ),
    ]


def _build_retry_hint(validation_result: dict | None) -> RetryHint:
    """validator_node가 남긴 failure_code/retryable을 supervisor의 retry_hint로 변환한다.

    retryable=True인 경우에만 supervisor가 eda_agent를 통째로 1회 더 호출한다
    (supervisor/validation.py의 기존 재시도 엔진 재사용, 그 외 필드는 기본값 유지)."""
    validation_result = validation_result or {}
    if not validation_result.get("retryable"):
        return RetryHint()
    return RetryHint(
        retryable=True,
        reason_code="eda_self_validation_failed_retryable",
        suggested_action="rerun_eda_agent",
        details={
            "failure_code": validation_result.get("failure_code", ""),
            "failure_reason": validation_result.get("reason", ""),
        },
    )

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from data_agent_backend.models.artifacts import ArtifactType

from DATA_Analyst_Assistant_Agent.agents.artifact_data import CsvArtifactData, read_sql_result_csvs
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.agents.eda._runtime import EdaContext, reset_context, set_context
from DATA_Analyst_Assistant_Agent.shared.contracts import AgentEnvelope, LocalCheck, OrchestrationState, ValidationBlock


class EDAAgent:
    name = "eda_agent"

    def run(self, state: OrchestrationState, runtime: AgentRuntime) -> AgentEnvelope:
        context = runtime.context(state, node_name=self.name, tool_name="eda_agent.lang_graph")
        csvs = read_sql_result_csvs(state, runtime)
        source_ids = [csv.artifact_id for csv in csvs]
        profile = profile_from_csv_artifacts(csvs)

        # 원본 LangGraph EDA 실행 (planner → 분석 노드 → insight/hypothesis → chart_selector)
        eda_result = self._run_eda_graph(csvs, state)

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
            "key_charts": eda_result.get("key_charts", []),
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
            artifact_refs=[ref],
            validation=ValidationBlock(local_checks=run_eda_self_check(source_ids, profile)),
            # 서브에이전트는 핸드오프를 갖지 않는다 — 다음 단계 라우팅은 메인(supervisor)의 몫.
            # 빈 값으로 명시(공용 기본값 "validation_agent"가 삭제된 에이전트라 폴백 방지).
            next_handoff="",
        )

    def _run_eda_graph(self, csvs: list[Any], state: OrchestrationState) -> dict[str, Any]:
        """CSV 아티팩트로 DataFrame을 만들고 원본 EDA LangGraph를 실행한다."""
        from DATA_Analyst_Assistant_Agent.agents.eda.graph import build_app

        frames = [csv.dataframe for csv in csvs if csv.error is None and not csv.dataframe.empty]
        if not frames:
            raise RuntimeError("No non-empty SQL CSV artifact was available for EDA.")
        df = pd.concat(frames, ignore_index=True)

        # 원본 모듈 전역(_df 등)을 대체하는 실행 컨텍스트. df 만 채우고
        # key/measure/time 컬럼은 load_mart 노드가 확정한다.
        set_context(EdaContext(df=df, question_type=state.route_kind or ""))
        # 앞단이 넘긴 의미 힌트를 EDA로 전달(있으면 줍고 없으면 폴백). 앞단(오케스트레이터)은
        # "매출"/"월"/None 수준이라 대개 비어 옴 → 가설 노드가 priority_metrics로 폴백한다.
        plan = state.plan
        plan_metric = (plan.metric if plan and plan.metric else "") or ""
        plan_dimension = (plan.dimension if plan and plan.dimension else "") or ""
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


def run_eda_self_check(source_artifact_ids: list[str], profile: dict) -> list[LocalCheck]:
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
    ]

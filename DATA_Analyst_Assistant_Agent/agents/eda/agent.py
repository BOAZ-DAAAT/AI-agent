from __future__ import annotations

import json
from typing import Any

import pandas as pd

from data_agent_backend.models.artifacts import ArtifactType

from DATA_Analyst_Assistant_Agent.agents.artifact_data import read_sql_result_csvs
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.agents.eda._runtime import EdaContext, reset_context, set_context
from DATA_Analyst_Assistant_Agent.agents.eda.profiler import profile_from_csv_artifacts
from DATA_Analyst_Assistant_Agent.agents.eda.self_check import run_eda_self_check
from DATA_Analyst_Assistant_Agent.shared.contracts import AgentEnvelope, OrchestrationState, ValidationBlock


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
            next_handoff="validation_agent",
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
        # 앞단이 넘긴 의미 힌트를 EDA로 전달(있으면 줍고 없으면 폴백). 현재 supervisor는
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

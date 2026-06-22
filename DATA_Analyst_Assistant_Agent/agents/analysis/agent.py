from __future__ import annotations

import json
from typing import Any

from data_agent_backend.models.artifacts import ArtifactType

from DATA_Analyst_Assistant_Agent.agents.artifact_data import first_dataframe, read_json_artifact, read_sql_result_csvs
from DATA_Analyst_Assistant_Agent.agents.analysis.workflow import run_analysis_workflow
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.shared.contracts import (
    AgentEnvelope,
    ApprovalRequirement,
    OrchestrationState,
    ValidationBlock,
)


class AnalysisAgent:
    name = "analysis_agent"

    def run(
        self,
        state: OrchestrationState,
        runtime: AgentRuntime,
        *,
        question_type: str | None = None,
        planner_model: Any | None = None,
    ) -> AgentEnvelope:
        context = runtime.context(state, node_name=self.name, tool_name="analysis_agent.result")
        parent_ids = state.artifact_ids.get("eda_agent", []) + state.artifact_ids.get("sql_agent", [])
        eda_profiles = []
        for artifact_id in state.artifact_ids.get("eda_agent", []):
            payload = read_json_artifact(runtime, artifact_id)
            eda_profiles.append(payload.get("profile", payload))
        csvs = read_sql_result_csvs(state, runtime)
        result, local_checks, terminal_reason = run_analysis_workflow(
            state,
            first_dataframe(csvs),
            eda_profiles,
            question_type=question_type,
            planner_model=planner_model,
        )
        ref = runtime.adapter.register_artifact(
            state.run_id,
            ArtifactType.file,
            content_text=json.dumps(result, ensure_ascii=False, indent=2),
            filename="analysis_result.json",
            created_by_tool="DATA_Analyst_Assistant_Agent.analysis",
            context=context,
            parent_ids=parent_ids,
            lineage_edge_type="derived_from",
            metadata={
                "kind": "analysis_result",
                "source_artifact_count": len(parent_ids),
                "terminal_reason": terminal_reason,
            },
            preview={
                "method_summary": result["method_summary"],
                "key_findings": result["key_findings"],
                "limitations": result["limitations"],
                "data_quality_notes": result["data_quality_notes"],
                "analysis_kind": result["plan"]["analysis_kind"],
                "tool_names": result["plan"]["tool_names"],
                "human_review": result["human_review"],
            },
        )
        review = result["human_review"]
        return AgentEnvelope(
            agent_name=self.name,
            summary="Analysis result generated from SQL result CSV and EDA profile artifacts.",
            artifact_refs=[ref],
            validation=ValidationBlock(local_checks=local_checks),
            approval=ApprovalRequirement(
                required=review["required"],
                reason=review["reason"],
                approval_type="analysis.review" if review["required"] else "",
            ),
            next_handoff="validation_agent",
        )

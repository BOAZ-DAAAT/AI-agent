from __future__ import annotations

import json
from typing import Any

from data_agent_backend.models.artifacts import ArtifactType

from DATA_Analyst_Assistant_Agent.agents.artifact_data import first_dataframe, load_analysis_inputs, read_json_artifact
from DATA_Analyst_Assistant_Agent.agents.analysis.graph import run_analysis_workflow
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.shared.contracts import (
    AgentEnvelope,
    AgentStatus,
    ApprovalRequirement,
    OrchestrationState,
    RetryHint,
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
        code_generator_model: Any | None = None,
        critic_model: Any | None = None,
        chart_artifact_loader: Any | None = None,
        chart_reader: Any | None = None,
    ) -> AgentEnvelope:
        context = runtime.context(state, node_name=self.name, tool_name="analysis_agent.result")
        parent_ids = state.artifact_ids.get("eda_agent", []) + state.artifact_ids.get("sql_agent", [])
        eda_profiles = []
        for artifact_id in state.artifact_ids.get("eda_agent", []):
            payload = read_json_artifact(runtime, artifact_id)
            eda_profiles.append(payload)
        csvs = load_analysis_inputs(state, runtime)
        result, local_checks, terminal_reason = run_analysis_workflow(
            state,
            first_dataframe(csvs),
            eda_profiles,
            question_type=question_type,
            planner_model=planner_model,
            code_generator_model=code_generator_model,
            critic_model=critic_model,
            chart_artifact_loader=chart_artifact_loader,
            chart_reader=chart_reader,
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
        workflow_failed = terminal_reason != "validated_result"
        codegen_attempts = int(result.get("codegen_attempts") or 0)
        failure_reason = _analysis_failure_reason(result, terminal_reason)
        return AgentEnvelope(
            status=AgentStatus.failed if workflow_failed else AgentStatus.success,
            agent_name=self.name,
            summary=(
                f"Analysis workflow did not pass validation: {failure_reason}"
                if workflow_failed
                else "Analysis result generated from SQL result CSV and EDA profile artifacts."
            ),
            artifact_refs=[ref],
            validation=ValidationBlock(local_checks=local_checks),
            retry_hint=RetryHint(
                retryable=workflow_failed,
                suggested_action="retry_analysis" if workflow_failed else "continue",
                reason_code=terminal_reason if workflow_failed else "none",
                details={
                    "terminal_reason": terminal_reason,
                    "failure_reason": failure_reason,
                    "codegen_attempts": codegen_attempts,
                    "agent_retry_budget": state.max_retry_per_agent,
                } if workflow_failed else {},
            ),
            approval=ApprovalRequirement(
                required=(not workflow_failed) and review["required"],
                reason=review["reason"] if (not workflow_failed) and review["required"] else "",
                approval_type="analysis.review" if (not workflow_failed) and review["required"] else "",
            ),
            error=failure_reason if workflow_failed else "",
        )


def _analysis_failure_reason(result: dict[str, Any], terminal_reason: str) -> str:
    critique = result.get("code_critique")
    if isinstance(critique, dict):
        feedback = str(critique.get("feedback") or "").strip()
        if feedback:
            return feedback
        method_issues = [str(item).strip() for item in critique.get("method_issues", []) if str(item).strip()]
        if method_issues:
            return "; ".join(method_issues)

    limitations = [str(item).strip() for item in result.get("limitations", []) if str(item).strip()]
    for limitation in limitations:
        if "Analysis did not pass method review" in limitation:
            return limitation
    if limitations:
        return limitations[0]
    return terminal_reason

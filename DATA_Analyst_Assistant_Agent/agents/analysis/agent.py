from __future__ import annotations

import json
from typing import Any

from data_agent_backend.models.artifacts import ArtifactType

from DATA_Analyst_Assistant_Agent.agents.artifact_data import first_dataframe, load_analysis_inputs, read_json_artifact
from DATA_Analyst_Assistant_Agent.agents.analysis.graph import run_analysis_workflow
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisSelectionResponse,
    ReviewRequest,
)
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
        selection_response: AnalysisSelectionResponse | None = None,
        review_request: ReviewRequest | None = None,
    ) -> AgentEnvelope:
        context = runtime.context(state, node_name=self.name, tool_name="analysis_agent.result")
        _emit_progress(runtime, state, "workflow", "started", 0)
        parent_ids = state.artifact_ids.get("eda_agent", []) + state.artifact_ids.get("sql_agent", [])
        eda_profiles = []
        for artifact_id in state.artifact_ids.get("eda_agent", []):
            if not _is_json_profile_artifact(runtime, artifact_id):
                continue
            try:
                payload = read_json_artifact(runtime, artifact_id)
            except Exception:
                continue
            eda_profiles.append(payload)
        csvs = load_analysis_inputs(state, runtime)
        _emit_progress(runtime, state, "inputs", "loaded", 0)
        result, local_checks, terminal_reason = run_analysis_workflow(
            state,
            first_dataframe(csvs),
            eda_profiles,
            question_type=question_type,
            selection_response=selection_response,
            review_request=review_request,
            planner_model=planner_model,
            code_generator_model=code_generator_model,
            critic_model=critic_model,
            chart_artifact_loader=chart_artifact_loader,
            chart_reader=chart_reader,
            progress_callback=lambda stage, status, attempt: _emit_progress(
                runtime, state, stage, status, attempt
            ),
        )
        _emit_progress(runtime, state, "workflow", "completed", int(result.get("codegen_attempts") or 0))
        debug_payload = _debug_payload(result, terminal_reason)
        debug_ref = runtime.adapter.register_artifact(
            state.run_id,
            ArtifactType.file,
            content_text=json.dumps(debug_payload, ensure_ascii=False, indent=2),
            filename="analysis_debug.json",
            created_by_tool="DATA_Analyst_Assistant_Agent.analysis.debug",
            context=context,
            parent_ids=parent_ids,
            lineage_edge_type="derived_from",
            metadata={
                "kind": "analysis_debug",
                "source_artifact_count": len(parent_ids),
                "terminal_reason": terminal_reason,
            },
            preview={
                "terminal_reason": terminal_reason,
                "codegen_attempts": debug_payload.get("codegen_attempts", 0),
                "critique_verdict": (debug_payload.get("code_critique") or {}).get("verdict"),
            },
        )

        public_result = _public_result_payload(result, debug_ref.artifact_id)
        ref = runtime.adapter.register_artifact(
            state.run_id,
            ArtifactType.file,
            content_text=json.dumps(public_result, ensure_ascii=False, indent=2),
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
                "status": public_result.get("status"),
                "status_label": _status_label(public_result.get("status")),
                "title": public_result.get("title"),
                "executive_summary": public_result.get("executive_summary"),
                "key_findings": public_result.get("key_findings", [])[:5],
                "limitations": public_result.get("limitations", [])[:5],
                "method_notes": public_result.get("method_notes", [])[:5],
                "top_evidence_tables": public_result.get("evidence_tables", [])[:2],
                "review_request": _review_request_preview(public_result.get("review_request")),
                "human_review": public_result["human_review"],
            },
        )
        review = public_result["human_review"]
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
            artifact_refs=[ref, debug_ref],
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


def _emit_progress(
    runtime: AgentRuntime,
    state: OrchestrationState,
    stage: str,
    status: str,
    attempt: int,
) -> None:
    messages = {
        ("workflow", "started"): "analysis_agent execution started",
        ("inputs", "loaded"): "analysis input artifacts loaded",
        ("generate", "started"): "analysis code generation started",
        ("generate", "completed"): "analysis code generation completed",
        ("execute", "started"): "generated analysis code execution started",
        ("execute", "completed"): "generated analysis code execution completed",
        ("critic", "started"): "analysis method review started",
        ("critic", "completed"): "analysis method review completed",
        ("workflow", "completed"): "analysis workflow completed",
    }
    message = messages.get((stage, status), f"analysis {stage}: {status}")
    try:
        runtime.adapter.append_run_event(
            state.run_id,
            "analysis.progress",
            message,
            node_name="analysis_agent",
            metadata={"stage": stage, "status": status, "attempt": attempt},
        )
    except Exception:
        # Telemetry is auxiliary. Do not fail a completed analysis when event storage is unavailable.
        pass


def _is_json_profile_artifact(runtime: AgentRuntime, artifact_id: str) -> bool:
    try:
        artifact = runtime.adapter.get_artifact(artifact_id)
    except Exception:
        return False
    metadata = getattr(artifact, "metadata", None) or {}
    filename = str(getattr(artifact, "filename", "") or "").lower()
    artifact_type = str(getattr(artifact, "type", "") or "")
    if metadata.get("kind") == "eda_summary":
        return True
    if filename.endswith(".json"):
        return True
    return artifact_type.endswith("data_profile")


def _debug_payload(result: dict[str, Any], terminal_reason: str) -> dict[str, Any]:
    return {
        "run_id": result.get("run_id"),
        "status": result.get("status"),
        "terminal_reason": terminal_reason,
        "generated_code": result.get("generated_code", ""),
        "code_critique": result.get("code_critique"),
        "codegen_attempts": result.get("codegen_attempts", 0),
        "answer_coverage": result.get("answer_coverage"),
        "raw_statistics": [
            evidence.get("statistics", {})
            for evidence in result.get("evidence", [])
            if isinstance(evidence, dict)
        ],
        "source_artifacts": result.get("source_artifacts", {}),
        "hypothesis_tests": result.get("hypothesis_tests", []),
        "review_request": result.get("review_request"),
        "method_notes": result.get("method_notes", []),
        "method_decision": result.get("method_decision"),
    }


def _public_result_payload(result: dict[str, Any], debug_artifact_id: str) -> dict[str, Any]:
    payload = dict(result)
    payload["debug_artifact_id"] = debug_artifact_id
    payload["generated_code"] = ""
    payload["code_critique"] = None
    return payload


def _status_label(status: object) -> str:
    if status == "success":
        return "Analysis completed"
    if status == "review_required":
        return "Analysis completed; analysis decision review required"
    if status == "failed":
        return "Analysis failed"
    return "Analysis status unknown"


def _review_request_preview(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return ReviewRequest.model_validate(value).model_dump(mode="json")

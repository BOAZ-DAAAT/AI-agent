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
    ValidationFinding,
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
            progress_callback=lambda stage, status, attempt, metadata=None: _emit_progress(
                runtime, state, stage, status, attempt, metadata
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
        envelope_local_checks = [
            check.model_copy(update={"severity": "warning"})
            if workflow_failed and check.severity == "error"
            else check
            for check in local_checks
        ]
        return AgentEnvelope(
            status=AgentStatus.warning if workflow_failed else AgentStatus.success,
            agent_name=self.name,
            summary=(
                "분석이 검증을 완전히 통과하지 못했지만 결과를 보존했습니다: "
                f"{failure_reason}"
                if workflow_failed
                else "Analysis result generated from SQL result CSV and EDA profile artifacts."
            ),
            artifact_refs=[ref, debug_ref],
            validation=ValidationBlock(
                local_checks=envelope_local_checks,
                findings=_analysis_validation_findings(
                    public_result,
                    terminal_reason,
                    workflow_failed,
                ),
            ),
            retry_hint=RetryHint(
                retryable=False,
                suggested_action="continue",
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
            error="",
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
        if (
            "Analysis did not pass method review" in limitation
            or "Analysis code failed at the" in limitation
        ):
            return limitation
    if limitations:
        return limitations[0]
    return terminal_reason


def _analysis_validation_findings(
    public_result: dict[str, Any],
    terminal_reason: str,
    workflow_failed: bool,
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    if workflow_failed:
        findings.append(
            ValidationFinding(
                code=terminal_reason,
                source="analysis_workflow",
                severity="warning",
                disposition="limitation",
                message="분석 워크플로가 검증된 결과를 생성하지 못했습니다.",
                retryable=False,
                suggested_action="continue",
                details={
                    "status": public_result.get("status"),
                    "terminal_reason": terminal_reason,
                },
            )
        )

    review_request = public_result.get("review_request")
    if not workflow_failed and isinstance(review_request, dict):
        findings.append(
            ValidationFinding(
                code="analysis_review_request",
                source="analysis_workflow",
                severity="info",
                disposition="semantic_evidence",
                message=str(review_request.get("question") or "Analysis result includes a review request."),
                retryable=False,
                suggested_action="request_human_review",
                details={
                    "decision_type": review_request.get("decision_type"),
                    "recommended_option_id": review_request.get("recommended_option_id"),
                },
            )
        )

    for index, note in enumerate(public_result.get("method_notes") or []):
        text = str(note).strip()
        if not text:
            continue
        findings.append(
            ValidationFinding(
                code="analysis_method_note",
                source="analysis_critic",
                severity="warning",
                disposition="limitation",
                message=text,
                retryable=False,
                details={"index": index},
            )
        )

    for index, limitation in enumerate(public_result.get("limitations") or []):
        text = str(limitation).strip()
        if not text:
            continue
        findings.append(
            ValidationFinding(
                code="analysis_limitation",
                source="analysis_result",
                severity="warning",
                disposition="limitation",
                message=text,
                retryable=False,
                details={"index": index},
            )
        )
    return findings


def _emit_progress(
    runtime: AgentRuntime,
    state: OrchestrationState,
    stage: str,
    status: str,
    attempt: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    messages = {
        ("workflow", "started"): "analysis_agent execution started",
        ("inputs", "loaded"): "analysis input artifacts loaded",
        ("generate", "started"): "analysis code generation started",
        ("generate", "completed"): "analysis code generation completed",
        ("execute.preflight", "started"): "analysis generated code preflight started",
        ("execute.preflight", "completed"): "analysis generated code preflight completed",
        ("execute", "started"): "generated analysis code execution started",
        ("execute", "completed"): "generated analysis code execution completed",
        ("execute", "skipped_manual_run"): "generated analysis code execution skipped for manual run",
        ("contract_check", "started"): "analysis result contract check started",
        ("contract_check", "completed"): "analysis result contract check completed",
        ("critic", "started"): "analysis method review started",
        ("critic", "completed"): "analysis method review completed",
        ("critic", "skipped_precheck"): "analysis method review skipped by deterministic precheck",
        ("workflow", "completed"): "analysis workflow completed",
    }
    message = messages.get((stage, status), f"analysis {stage}: {status}")
    event_metadata = {
        "stage": stage,
        "status": status,
        "attempt": attempt,
        **_progress_metadata(runtime, state, stage, status, attempt, metadata or {}),
    }
    try:
        runtime.adapter.append_run_event(
            state.run_id,
            "analysis.progress",
            message,
            node_name="analysis_agent",
            metadata=event_metadata,
        )
    except Exception:
        # Telemetry is auxiliary. Do not fail a completed analysis when event storage is unavailable.
        pass


def _progress_metadata(
    runtime: AgentRuntime,
    state: OrchestrationState,
    stage: str,
    status: str,
    attempt: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if stage == "generate" and status == "completed" and metadata.get("generated_code"):
        artifact_id = _save_generated_code_artifact(runtime, state, attempt, metadata)
        return {
            "generated_code_artifact_id": artifact_id,
            "generated_code_length": len(str(metadata.get("generated_code") or "")),
        } if artifact_id else {"generated_code_length": len(str(metadata.get("generated_code") or ""))}
    return dict(metadata)


def _save_generated_code_artifact(
    runtime: AgentRuntime,
    state: OrchestrationState,
    attempt: int,
    metadata: dict[str, Any],
) -> str:
    code = str(metadata.get("generated_code") or "")
    imports = str(metadata.get("generated_imports") or "")
    source = f"{imports}\n{code}" if imports else code
    if not source.strip():
        return ""
    try:
        context = runtime.context(state, node_name="analysis_agent", tool_name="analysis_agent.generated_code")
        ref = runtime.adapter.register_artifact(
            state.run_id,
            ArtifactType.file,
            content_text=source,
            filename=f"analysis_generated_code_attempt_{attempt}.py",
            created_by_tool="DATA_Analyst_Assistant_Agent.analysis.generated_code",
            context=context,
            parent_ids=state.artifact_ids.get("sql_agent", []) + state.artifact_ids.get("eda_agent", []),
            lineage_edge_type="derived_from",
            metadata={
                "kind": "analysis_generated_code",
                "attempt": attempt,
                "rationale": str(metadata.get("generated_rationale") or ""),
            },
            preview={
                "attempt": attempt,
                "code_length": len(source),
                "rationale": str(metadata.get("generated_rationale") or "")[:500],
            },
        )
    except Exception:
        return ""
    return ref.artifact_id


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
        "error_history": result.get("error_history", []),
    }


def _public_result_payload(result: dict[str, Any], debug_artifact_id: str) -> dict[str, Any]:
    payload = dict(result)
    payload["debug_artifact_id"] = debug_artifact_id
    payload["code_critique"] = None
    payload["error_history"] = []
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

from __future__ import annotations

import json
from typing import Any

from data_agent_backend.models.artifacts import ArtifactType

from DATA_Analyst_Assistant_Agent.agents.artifact_data import generated_sql_from_artifacts, read_json_artifact, sql_result_artifact_ids
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.agents.sql.self_check import is_sql_safe
from DATA_Analyst_Assistant_Agent.shared.contracts import (
    AgentEnvelope,
    AgentStatus,
    BusinessFlag,
    LocalCheck,
    OrchestrationState,
    RetryHint,
    ValidationBlock,
)


class CentralValidationAgent:
    name = "validation_agent"

    def run(self, state: OrchestrationState, runtime: AgentRuntime, upstream: AgentEnvelope) -> AgentEnvelope:
        context = runtime.context(state, node_name=self.name, tool_name="validation_agent.business_context")
        findings: list[dict[str, Any]] = []
        approval = upstream.approval

        if upstream.validation.has_errors:
            findings.append(
                {
                    "category": "artifact_dependency_mismatch",
                    "severity": "error",
                    "retryable": True,
                    "detail": "Upstream local check failed.",
                }
            )
        elif upstream.approval.required:
            findings.append(
                {
                    "category": "approval_required_for_persistence",
                    "severity": "info",
                    "retryable": False,
                    "detail": upstream.approval.reason,
                }
            )
        elif upstream.validation.has_warnings:
            findings.append(
                {
                    "category": "business_context_warning",
                    "severity": "warning",
                    "retryable": False,
                    "detail": "Upstream result contains warnings.",
                }
            )
        else:
            findings.append(
                {
                    "category": "business_context_pass",
                    "severity": "info",
                    "retryable": False,
                    "detail": "Upstream result can continue.",
                }
            )

        findings.extend(self._artifact_existence_findings(state, runtime, upstream))
        findings.extend(self._sql_findings(state, runtime))
        findings.extend(self._eda_findings(state, runtime))
        findings.extend(self._report_evidence_findings(state, runtime, upstream))

        verdict, status, retry_hint = self._verdict(findings, approval_required=upstream.approval.required)

        validation_ref = runtime.adapter.register_artifact(
            state.run_id,
            ArtifactType.file,
            content_text=json.dumps(
                {
                    "run_id": state.run_id,
                    "upstream_agent": upstream.agent_name,
                    "verdict": verdict,
                    "goal": state.goal,
                    "findings": findings,
                    "artifact_ids": upstream.artifact_ids(),
                    "retryable": retry_hint.retryable,
                    "suggested_action": retry_hint.suggested_action,
                },
                ensure_ascii=False,
                indent=2,
            ),
            filename=f"central_validation_{upstream.agent_name}.json",
            created_by_tool="DATA_Analyst_Assistant_Agent.central_validation",
            context=context,
            parent_ids=upstream.artifact_ids(),
            lineage_edge_type="validates_business_context",
            metadata={"kind": "central_business_validation", "upstream_agent": upstream.agent_name, "verdict": verdict},
            preview={"verdict": verdict, "upstream_agent": upstream.agent_name, "finding_count": len(findings)},
        )

        return AgentEnvelope(
            status=status,
            agent_name=self.name,
            summary=f"Central validation verdict: {verdict}.",
            artifact_refs=[validation_ref],
            validation=ValidationBlock(
                local_checks=[
                    LocalCheck(
                        name="business_context_verdict",
                        passed=verdict in {"pass", "warn", "approval_required"},
                        severity="error" if verdict == "fail" else "info",
                        detail=verdict,
                    )
                ],
                business_flags=[
                    BusinessFlag(code=item["category"], severity=item.get("severity", "info"), message=item["detail"])
                    for item in findings
                ],
            ),
            retry_hint=retry_hint,
            approval=approval,
            context_refs=[{"kind": "upstream_agent", "ref_id": upstream.agent_name, "summary": upstream.summary}],
            next_handoff="supervisor_gate",
        )

    def _artifact_existence_findings(
        self, state: OrchestrationState, runtime: AgentRuntime, upstream: AgentEnvelope,
    ) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for artifact_id in upstream.artifact_ids():
            try:
                runtime.adapter.get_artifact(artifact_id)
                runtime.adapter.read_artifact_text(artifact_id)
            except Exception as exc:
                findings.append(
                    {
                        "category": "artifact_missing",
                        "severity": "error",
                        "retryable": False,
                        "detail": f"{artifact_id} could not be read: {exc}",
                    }
                )
        return findings

    def _sql_findings(self, state: OrchestrationState, runtime: AgentRuntime) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        if not state.artifact_ids.get("sql_agent"):
            return findings
        generated_sql = generated_sql_from_artifacts(state, runtime)
        if generated_sql and not is_sql_safe(generated_sql):
            findings.append(
                {
                    "category": "unsafe_sql",
                    "severity": "error",
                    "retryable": True,
                    "detail": "Generated SQL did not pass SELECT/WITH safety validation.",
                }
            )
        for artifact_id in sql_result_artifact_ids(state, runtime):
            artifact = runtime.adapter.get_artifact(artifact_id)
            row_count = int((artifact.preview or {}).get("row_count", 0) or 0)
            if row_count <= 0:
                findings.append(
                    {
                        "category": "empty_sql_result",
                        "severity": "warning",
                        "retryable": True,
                        "detail": f"{artifact_id} returned zero rows.",
                    }
                )
        return findings

    def _eda_findings(self, state: OrchestrationState, runtime: AgentRuntime) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for artifact_id in state.artifact_ids.get("eda_agent", []):
            try:
                payload = read_json_artifact(runtime, artifact_id)
            except Exception as exc:
                findings.append(
                    {
                        "category": "eda_profile_unreadable",
                        "severity": "error",
                        "retryable": True,
                        "detail": f"{artifact_id} could not be read: {exc}",
                    }
                )
                continue
            profile = payload.get("profile", payload)
            quality_status = profile.get("quality_status")
            if quality_status in {"unavailable", "needs_review"}:
                findings.append(
                    {
                        "category": "eda_quality_status",
                        "severity": "warning" if quality_status == "needs_review" else "error",
                        "retryable": True,
                        "detail": f"{artifact_id} quality_status={quality_status}.",
                    }
                )
        return findings

    def _report_evidence_findings(
        self, state: OrchestrationState, runtime: AgentRuntime, upstream: AgentEnvelope,
    ) -> list[dict[str, Any]]:
        if upstream.agent_name != "report_agent":
            return []
        report_ids = upstream.artifact_ids()
        if not report_ids:
            return [
                {
                    "category": "report_missing",
                    "severity": "error",
                    "retryable": True,
                    "detail": "Report agent did not register a report artifact.",
                }
            ]
        report_text = runtime.adapter.read_artifact_text(report_ids[0])
        expected_ids = [
            artifact_id
            for agent_name, artifact_ids in state.artifact_ids.items()
            if agent_name != "report_agent"
            for artifact_id in artifact_ids
        ]
        missing = [artifact_id for artifact_id in expected_ids if artifact_id not in report_text]
        if not missing:
            return []
        return [
            {
                "category": "report_evidence_missing",
                "severity": "error",
                "retryable": True,
                "detail": "Report omitted upstream evidence ids: " + ", ".join(missing),
            }
        ]

    def _verdict(
        self, findings: list[dict[str, Any]], *, approval_required: bool,
    ) -> tuple[str, AgentStatus, RetryHint]:
        errors = [item for item in findings if item.get("severity") == "error"]
        warnings = [item for item in findings if item.get("severity") == "warning"]
        retryable = any(item.get("retryable") for item in errors)
        if errors:
            return (
                "fail",
                AgentStatus.failed,
                RetryHint(
                    retryable=retryable,
                    suggested_action="retry_upstream" if retryable else "manual_review",
                    reason_code=errors[0]["category"],
                ),
            )
        if approval_required:
            return "approval_required", AgentStatus.approval_required, RetryHint()
        if warnings:
            retryable_warning = any(item.get("retryable") for item in warnings)
            return (
                "warn",
                AgentStatus.warning,
                RetryHint(
                    retryable=retryable_warning,
                    suggested_action="review_or_retry" if retryable_warning else "continue",
                    reason_code=warnings[0]["category"],
                ),
            )
        return "pass", AgentStatus.success, RetryHint()

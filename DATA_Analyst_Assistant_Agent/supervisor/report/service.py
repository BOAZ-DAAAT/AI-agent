from __future__ import annotations

import hashlib
from typing import Any

from data_agent_backend.models.artifacts import ArtifactRecord, ArtifactType

from DATA_Analyst_Assistant_Agent.agents.artifact_data import generated_sql_from_artifacts, read_json_artifact
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.supervisor.report.builder import build_report
from DATA_Analyst_Assistant_Agent.supervisor.report.self_check import run_report_self_check
from DATA_Analyst_Assistant_Agent.shared.contracts import (
    AgentEnvelope,
    AgentStatus,
    OrchestrationState,
    ValidationBlock,
)


REPORT_TOOL_NAME = "report_agent.final_report"
REPORT_ARTIFACT_KIND = "final_report"


def generate_report_envelope(
    state: OrchestrationState,
    runtime: AgentRuntime,
    *,
    node_name: str,
) -> AgentEnvelope:
    parent_ids = _source_artifact_ids(state)
    eda_profile = _eda_profile(state, runtime)
    analysis_result = _analysis_result(state, runtime)
    report = build_report(
        state,
        generated_sql=generated_sql_from_artifacts(state, runtime),
        eda_profile=eda_profile,
        analysis_result=analysis_result,
    )
    checks = run_report_self_check(report)
    validation = ValidationBlock(local_checks=checks)
    if any(not check.passed for check in checks):
        return AgentEnvelope(
            status=AgentStatus.failed,
            agent_name="report_agent",
            summary="보고서 자체 검사에 실패했습니다.",
            validation=validation,
            error="보고서 자체 검사에 실패했습니다.",
        )

    content_hash = hashlib.sha256(report.encode("utf-8")).hexdigest()
    existing = _matching_report_artifact(runtime, state.run_id, content_hash)
    if existing is not None:
        ref = existing.ref()
    else:
        context = runtime.context(state, node_name=node_name, tool_name=REPORT_TOOL_NAME)
        ref = runtime.adapter.register_artifact(
            state.run_id,
            ArtifactType.report,
            content_text=report,
            created_by_tool=REPORT_TOOL_NAME,
            context=context,
            filename="final_report.md",
            metadata={
                "kind": REPORT_ARTIFACT_KIND,
                "source_artifact_count": len(parent_ids),
                "source_artifact_ids": parent_ids,
            },
            parent_ids=parent_ids,
            preview={
                "report_sections": [
                    "summary",
                    "key_findings",
                    "evidence",
                    "visuals",
                    "limitations",
                    "next_actions",
                ]
            },
        )
    return AgentEnvelope(
        agent_name="report_agent",
        summary="Final report generated from registered evidence artifacts.",
        artifact_refs=[ref],
        validation=validation,
    )


def _source_artifact_ids(state: OrchestrationState) -> list[str]:
    parent_ids: list[str] = []
    for agent_name, artifact_ids in state.artifact_ids.items():
        if agent_name == "report_agent":
            continue
        for artifact_id in artifact_ids:
            if artifact_id and artifact_id not in parent_ids:
                parent_ids.append(artifact_id)
    return parent_ids


def _eda_profile(state: OrchestrationState, runtime: AgentRuntime) -> dict[str, Any] | None:
    artifact_ids = state.artifact_ids.get("eda_agent", [])
    if not artifact_ids:
        return None
    payload = read_json_artifact(runtime, artifact_ids[0])
    profile = payload.get("profile", payload)
    return profile if isinstance(profile, dict) else None


def _analysis_result(state: OrchestrationState, runtime: AgentRuntime) -> dict[str, Any] | None:
    artifact_ids = state.artifact_ids.get("analysis_agent", [])
    if not artifact_ids:
        return None
    return read_json_artifact(runtime, artifact_ids[0])


def _matching_report_artifact(
    runtime: AgentRuntime,
    run_id: str,
    content_hash: str,
) -> ArtifactRecord | None:
    for artifact in runtime.adapter.list_artifacts(run_id=run_id, artifact_type=ArtifactType.report):
        if (
            artifact.metadata.get("kind") == REPORT_ARTIFACT_KIND
            and artifact.created_by_tool == REPORT_TOOL_NAME
            and artifact.content_hash == content_hash
        ):
            return artifact
    return None

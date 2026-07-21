from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from DATA_Analyst_Assistant_Agent.shared.contracts import ValidationFinding
from DATA_Analyst_Assistant_Agent.supervisor.capabilities import (
    DEFAULT_AGENT_CAPABILITIES,
    AgentCapability,
    EvidenceRequirement,
)
from DATA_Analyst_Assistant_Agent.supervisor.state import AgentCompactResult, SupervisorState


class EvidenceVerification(BaseModel):
    valid: bool
    decision: Literal["accept", "accept_with_limitations", "reject"]
    findings: list[ValidationFinding] = Field(default_factory=list)
    content_hashes: dict[str, str] = Field(default_factory=dict)


def verify_candidate_evidence(state: SupervisorState, backend_adapter: Any) -> EvidenceVerification:
    pending = state.get("pending_result")
    if not isinstance(pending, dict):
        return _rejected("pending_result_missing", "검증할 후보 결과가 없습니다.")
    try:
        result = AgentCompactResult.model_validate(pending.get("result") or {})
    except Exception as exc:
        return _rejected("candidate_schema_invalid", f"후보 결과 스키마가 올바르지 않습니다: {exc}")

    capability = _capability_for(result.agent)
    findings: list[ValidationFinding] = []
    records: dict[str, Any] = {}
    content_hashes: dict[str, str] = {}
    summaries = {artifact.artifact_id: artifact for artifact in result.artifacts}

    for artifact_id in dict.fromkeys([*result.artifact_ids, *summaries.keys()]):
        if not artifact_id:
            continue
        try:
            record = backend_adapter.get_artifact(artifact_id)
        except Exception as exc:
            findings.append(_finding("artifact_missing", f"아티팩트를 조회할 수 없습니다: {artifact_id}: {exc}"))
            continue
        records[artifact_id] = record
        if str(getattr(record, "run_id", "")) != str(state.get("current_run_id", "")):
            findings.append(_finding("cross_run_artifact", f"현재 run의 아티팩트가 아닙니다: {artifact_id}"))

        try:
            raw = backend_adapter.services.artifact_store.read_bytes(artifact_id)
        except Exception as exc:
            findings.append(_finding("artifact_unreadable", f"아티팩트를 읽을 수 없습니다: {artifact_id}: {exc}"))
            continue
        actual_hash = hashlib.sha256(raw).hexdigest()
        stored_hash = str(getattr(record, "content_hash", "") or "")
        content_hashes[artifact_id] = actual_hash
        if not stored_hash or stored_hash != actual_hash:
            findings.append(_finding("content_hash_invalid", f"저장된 content hash가 일치하지 않습니다: {artifact_id}"))
        summary = summaries.get(artifact_id)
        if summary is not None and summary.content_hash and summary.content_hash != actual_hash:
            findings.append(_finding("content_hash_mismatch", f"후보 content hash가 일치하지 않습니다: {artifact_id}"))
        if summary is not None:
            record_type = str(getattr(record, "type", ""))
            record_kind = str((getattr(record, "metadata", None) or {}).get("kind", ""))
            if summary.type and summary.type != record_type:
                findings.append(_finding("artifact_type_mismatch", f"아티팩트 type이 일치하지 않습니다: {artifact_id}"))
            if summary.kind and summary.kind != record_kind:
                findings.append(_finding("artifact_kind_mismatch", f"아티팩트 kind가 일치하지 않습니다: {artifact_id}"))

    for requirement in capability.output_evidence:
        matches = [record for record in records.values() if _matches(record, requirement)]
        if not matches:
            findings.append(
                _finding(
                    "required_evidence_missing",
                    f"필수 근거가 없습니다: type={requirement.type}, kind={requirement.kind}",
                )
            )
            continue
        for record in matches:
            _validate_content(record, requirement, backend_adapter, findings)

    _validate_lineage(result, records, state, findings)
    blocking = any(finding.disposition == "error" for finding in findings)
    limitations = any(finding.disposition in {"warning", "limitation"} for finding in findings)
    return EvidenceVerification(
        valid=not blocking,
        decision="reject" if blocking else "accept_with_limitations" if limitations else "accept",
        findings=findings,
        content_hashes=content_hashes,
    )


def _validate_content(
    record: Any,
    requirement: EvidenceRequirement,
    backend_adapter: Any,
    findings: list[ValidationFinding],
) -> None:
    artifact_id = str(record.artifact_id)
    try:
        text = backend_adapter.read_artifact_text(artifact_id)
    except Exception:
        return
    if requirement.non_empty and not text.strip():
        findings.append(_finding("empty_required_evidence", f"필수 근거가 비어 있습니다: {artifact_id}"))
    elif not text.strip():
        findings.append(
            ValidationFinding(
                code="empty_evidence",
                source="evidence_verifier",
                severity="warning",
                disposition="limitation",
                message=f"근거 결과가 비어 있습니다: {artifact_id}",
            )
        )

    if requirement.type == "sql_result":
        preview = getattr(record, "preview", None) or {}
        row_count = preview.get("row_count")
        if not isinstance(row_count, int) or row_count < 0:
            findings.append(_finding("sql_result_structure_invalid", f"SQL row_count가 올바르지 않습니다: {artifact_id}"))
    if requirement.kind in {"eda_summary", "analysis_result"}:
        try:
            payload = json.loads(text)
        except Exception:
            findings.append(_finding("json_evidence_invalid", f"JSON 근거를 파싱할 수 없습니다: {artifact_id}"))
            return
        metadata = getattr(record, "metadata", None) or {}
        terminal = payload.get("terminal_state") or payload.get("status") or metadata.get("terminal_state")
        if requirement.kind == "eda_summary" and terminal is None and not payload.get("error_log"):
            terminal = "success"
        if requirement.kind == "analysis_result" and terminal is None:
            terminal = metadata.get("terminal_reason")
        if terminal not in {"success", "completed", "succeeded", "validated_result"}:
            findings.append(_finding("terminal_metadata_invalid", f"성공 terminal metadata가 없습니다: {artifact_id}"))
    if requirement.type == "report" and not text.strip():
        findings.append(_finding("report_empty", f"Markdown 리포트가 비어 있습니다: {artifact_id}"))


def _validate_lineage(
    result: AgentCompactResult,
    records: dict[str, Any],
    state: SupervisorState,
    findings: list[ValidationFinding],
) -> None:
    accepted_ids = {
        str(item.get("artifact_id"))
        for items in state.get("accepted_evidence", {}).values()
        for item in items
        if item.get("artifact_id")
    }
    if result.agent == "sql_agent":
        query_ids = {
            artifact_id for artifact_id, record in records.items() if str(getattr(record, "type", "")) == "sql_query"
        }
        for artifact_id, record in records.items():
            if str(getattr(record, "type", "")) == "sql_result" and not query_ids.intersection(record.parent_ids):
                findings.append(_finding("lineage_missing", f"SQL 결과의 query parent가 없습니다: {artifact_id}"))
        return
    if result.agent in {"eda_agent", "analysis_agent", "report_agent"}:
        for artifact_id, record in records.items():
            if not accepted_ids.intersection(getattr(record, "parent_ids", []) or []):
                findings.append(_finding("lineage_missing", f"승격된 parent 근거가 없습니다: {artifact_id}"))


def _matches(record: Any, requirement: EvidenceRequirement) -> bool:
    return str(getattr(record, "type", "")) == requirement.type and str(
        (getattr(record, "metadata", None) or {}).get("kind", "")
    ) == requirement.kind


def _capability_for(agent: str) -> AgentCapability:
    return next(item for item in DEFAULT_AGENT_CAPABILITIES if item.agent == agent)


def _finding(code: str, message: str) -> ValidationFinding:
    return ValidationFinding(
        code=code,
        source="evidence_verifier",
        severity="error",
        disposition="error",
        message=message,
    )


def _rejected(code: str, message: str) -> EvidenceVerification:
    return EvidenceVerification(valid=False, decision="reject", findings=[_finding(code, message)])

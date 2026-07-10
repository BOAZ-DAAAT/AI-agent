from __future__ import annotations

import hashlib

import pytest
from data_agent_backend.config import BackendConfig

from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.supervisor.evidence import verify_candidate_evidence
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    ArtifactSummary,
    empty_supervisor_state,
    stage_candidate_result,
)


@pytest.fixture()
def adapter(tmp_path) -> BackendAdapter:
    return BackendAdapter(config=BackendConfig(base_data_dir=tmp_path / ".data_agent"))


def _state(run_id: str):
    return empty_supervisor_state(
        thread_id="thread_001",
        run_id=run_id,
        user_query="매출 요약",
        datasource_id=None,
    )


def _register_sql_pair(adapter: BackendAdapter, run_id: str):
    query = adapter.register_artifact(
        run_id,
        "sql_query",
        content_text="SELECT 1 AS value",
        filename="query.sql",
        created_by_tool="test",
        metadata={"kind": "generated_sql"},
    )
    result = adapter.register_artifact(
        run_id,
        "sql_result",
        content_text="value\r\n1\r\n",
        filename="result.csv",
        created_by_tool="test",
        parent_ids=[query.artifact_id],
        metadata={"kind": "sql_result"},
        preview={"row_count": 1, "columns": ["value"]},
    )
    return query, result


def test_valid_sql_evidence_passes_current_run_hash_type_and_lineage_checks(adapter: BackendAdapter) -> None:
    run = adapter.create_run()
    query, result = _register_sql_pair(adapter, run.run_id)
    candidate = AgentCompactResult(
        agent="sql_agent",
        status="success",
        summary="SQL 완료",
        artifact_ids=[query.artifact_id, result.artifact_id],
        artifacts=[
            ArtifactSummary(
                artifact_id=query.artifact_id,
                type="sql_query",
                kind="generated_sql",
                content_hash=query.content_hash,
            ),
            ArtifactSummary(
                artifact_id=result.artifact_id,
                type="sql_result",
                kind="sql_result",
                content_hash=result.content_hash,
                parent_ids=[query.artifact_id],
            ),
        ],
    )
    state = stage_candidate_result(_state(run.run_id), candidate, {"generated_sql": "SELECT 1 AS value"})

    verification = verify_candidate_evidence(state, adapter)

    assert verification.valid is True
    assert verification.decision == "accept"
    assert verification.content_hashes == {
        query.artifact_id: query.content_hash,
        result.artifact_id: result.content_hash,
    }


def test_missing_required_artifact_rejects_success_candidate(adapter: BackendAdapter) -> None:
    run = adapter.create_run()
    query, _ = _register_sql_pair(adapter, run.run_id)
    candidate = AgentCompactResult(
        agent="sql_agent",
        status="success",
        summary="결과 아티팩트 누락",
        artifact_ids=[query.artifact_id],
        artifacts=[ArtifactSummary(artifact_id=query.artifact_id, type="sql_query", kind="generated_sql")],
    )
    state = stage_candidate_result(_state(run.run_id), candidate, {})

    verification = verify_candidate_evidence(state, adapter)

    assert verification.valid is False
    assert verification.decision == "reject"
    assert any(finding.code == "required_evidence_missing" for finding in verification.findings)


def test_cross_run_and_hash_mismatch_are_rejected(adapter: BackendAdapter) -> None:
    current = adapter.create_run()
    other = adapter.create_run()
    query, result = _register_sql_pair(adapter, other.run_id)
    candidate = AgentCompactResult(
        agent="sql_agent",
        status="success",
        summary="다른 run 결과",
        artifact_ids=[query.artifact_id, result.artifact_id],
        artifacts=[
            ArtifactSummary(
                artifact_id=query.artifact_id,
                type="sql_query",
                kind="generated_sql",
                content_hash="0" * 64,
            ),
            ArtifactSummary(
                artifact_id=result.artifact_id,
                type="sql_result",
                kind="sql_result",
                content_hash=result.content_hash,
            ),
        ],
    )
    state = stage_candidate_result(_state(current.run_id), candidate, {})

    verification = verify_candidate_evidence(state, adapter)

    codes = {finding.code for finding in verification.findings}
    assert "cross_run_artifact" in codes
    assert "content_hash_mismatch" in codes


def test_unreadable_artifact_is_rejected(adapter: BackendAdapter) -> None:
    run = adapter.create_run()
    query, result = _register_sql_pair(adapter, run.run_id)
    adapter.services.artifact_store.get_path(result.artifact_id).unlink()
    candidate = AgentCompactResult(
        agent="sql_agent",
        status="success",
        summary="읽을 수 없는 결과",
        artifact_ids=[query.artifact_id, result.artifact_id],
        artifacts=[
            ArtifactSummary(artifact_id=query.artifact_id, type="sql_query", kind="generated_sql"),
            ArtifactSummary(artifact_id=result.artifact_id, type="sql_result", kind="sql_result"),
        ],
    )
    state = stage_candidate_result(_state(run.run_id), candidate, {})

    verification = verify_candidate_evidence(state, adapter)

    assert verification.valid is False
    assert any(finding.code == "artifact_unreadable" for finding in verification.findings)

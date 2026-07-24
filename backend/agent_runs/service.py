from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator
from uuid import uuid4

from sqlalchemy import inspect

from backend.config import StorageMySQL
from backend.session.schemas import SessionResponse
from data_agent_backend.models.common import BackendError
from data_agent_backend.models.runs import (
    TERMINAL_RUN_STATUSES,
    RunEvent,
    RunRecord,
    RunStatus,
)
from data_agent_backend.services.factory import BackendServices
from data_agent_backend.storage.filesystem import ensure_child_path

from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.shared.cancellation import (
    RunCancellationRequested,
    raise_if_run_cancelled,
)
from DATA_Analyst_Assistant_Agent.shared.contracts import SupervisorRunResult, SupervisorTerminalState
from DATA_Analyst_Assistant_Agent.supervisor.agent import SupervisorAgent
from DATA_Analyst_Assistant_Agent.supervisor.branch import BranchStage, branch_from
from DATA_Analyst_Assistant_Agent.supervisor.report.generator import generate_report
from DATA_Analyst_Assistant_Agent.supervisor.report.schemas import ReportResult
from DATA_Analyst_Assistant_Agent.supervisor.state import empty_supervisor_state, to_orchestration_state
from DATA_Analyst_Assistant_Agent.supervisor.summary.generator import generate_node_summary
from DATA_Analyst_Assistant_Agent.supervisor.summary.schemas import NodeSummaryResult


_SESSION_ENV_LOCK = threading.Lock()


class NodeSummaryNotFoundError(Exception):
    """완료 노드 또는 해당 노드의 상세 서머리를 찾지 못했을 때."""


class NodeReportGenerationError(Exception):
    """선택한 노드로 리포트를 생성할 수 없을 때."""


class RunDeletionConflictError(Exception):
    """아직 실행 중인 run 삭제를 요청했을 때."""


class RunCancellationConflictError(Exception):
    """이미 완료되거나 실패한 run의 취소를 요청했을 때."""


@dataclass(frozen=True)
class RunDeletionResult:
    run_id: str
    deleted_event_count: int
    deleted_artifact_count: int


@dataclass(frozen=True)
class RunCancellationResult:
    run_id: str
    discarded_node_id: str | None


_ACTIVE_NODE_EVENTS = {
    "agent.started",
    "agent.progress",
    "agent.retrying",
    "agent.waiting",
    "agent.resumed",
}
_TERMINAL_NODE_EVENTS = {"agent.completed", "agent.discarded", "agent.failed"}


def _active_node_event(events: list[RunEvent]) -> RunEvent | None:
    active: RunEvent | None = None
    for event in events:
        node_id = event.metadata.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            continue
        if event.event_type in _ACTIVE_NODE_EVENTS:
            active = event
        elif event.event_type in _TERMINAL_NODE_EVENTS and active is not None:
            if active.metadata.get("node_id") == node_id:
                active = None
    return active


def cancel_agent_run(
    *,
    services: BackendServices,
    run_id: str,
    cancelled_by: str,
) -> RunCancellationResult:
    try:
        services.run_service.cancel_run(
            run_id,
            metadata={"cancelled_by": cancelled_by, "cancel_reason": "user_requested"},
        )
    except BackendError as exc:
        if exc.code == "RUN_NOT_CANCELLABLE":
            raise RunCancellationConflictError(str(exc)) from exc
        raise

    active_event = _active_node_event(services.run_service.list_events(run_id))
    discarded_node_id: str | None = None
    if active_event is not None:
        discarded_node_id = str(active_event.metadata["node_id"])
        attempt = int(active_event.metadata.get("attempt") or 1)
        services.run_service.append_event(
            run_id,
            "agent.discarded",
            "사용자 요청으로 진행 중인 작업을 중단했습니다.",
            event_key=f"agent-node:{discarded_node_id}:agent.discarded:attempt:{attempt}",
            node_name=active_event.node_name,
            metadata={
                **active_event.metadata,
                "reason_code": "run_cancelled",
                "cancelled_by": cancelled_by,
            },
        )

    services.run_service.append_event(
        run_id,
        "run.cancelled",
        "사용자 요청으로 Agent 실행을 중단했습니다.",
        event_key=f"run:{run_id}:cancelled",
        node_name="supervisor",
        metadata={"cancelled_by": cancelled_by, "reason_code": "user_requested"},
    )
    return RunCancellationResult(run_id=run_id, discarded_node_id=discarded_node_id)


def delete_terminal_run_data(
    *,
    services: BackendServices,
    run_id: str,
) -> RunDeletionResult:
    run = services.run_service.get_run(run_id)
    if run.status not in TERMINAL_RUN_STATUSES:
        raise RunDeletionConflictError("완료되거나 실패한 실행만 삭제할 수 있습니다.")

    artifacts = services.artifact_registry.list_artifacts(run_id=run_id)
    artifact_ids = [artifact.artifact_id for artifact in artifacts]
    event_count_row = services.run_service.sqlite.query_one(
        "SELECT COUNT(*) AS count FROM run_events WHERE run_id = ?",
        (run_id,),
    )
    event_count = int(event_count_row["count"]) if event_count_row else 0

    with services.run_service.sqlite.connect() as conn:
        approval_rows = conn.execute(
            "SELECT approval_id FROM approval_requests WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        approval_ids = [row["approval_id"] for row in approval_rows]

        if approval_ids:
            placeholders = ",".join("?" for _ in approval_ids)
            conn.execute(
                f"DELETE FROM approval_events WHERE approval_id IN ({placeholders})",
                approval_ids,
            )
        if artifact_ids:
            placeholders = ",".join("?" for _ in artifact_ids)
            conn.execute(
                f"DELETE FROM exports WHERE artifact_id IN ({placeholders})",
                artifact_ids,
            )
            conn.execute(
                f"DELETE FROM claim_evidence WHERE artifact_id IN ({placeholders})",
                artifact_ids,
            )
            conn.execute(
                f"DELETE FROM artifact_lineage WHERE parent_id IN ({placeholders}) "
                f"OR child_id IN ({placeholders})",
                [*artifact_ids, *artifact_ids],
            )
            conn.execute(
                f"DELETE FROM artifact_previews WHERE artifact_id IN ({placeholders})",
                artifact_ids,
            )
            conn.execute(
                f"DELETE FROM artifacts WHERE artifact_id IN ({placeholders})",
                artifact_ids,
            )

        conn.execute("DELETE FROM approval_requests WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM run_events WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

    for artifact_id in artifact_ids:
        services.artifact_store.delete(artifact_id)

    if run.thread_id:
        remaining_thread_run = services.run_service.sqlite.query_one(
            "SELECT 1 FROM runs WHERE thread_id = ? LIMIT 1",
            (run.thread_id,),
        )
        if remaining_thread_run is None:
            for suffix in ("", "-wal", "-shm"):
                checkpoint_path = ensure_child_path(
                    services.config.base_data_dir,
                    services.config.base_data_dir / f"{run.thread_id}.sqlite{suffix}",
                )
                checkpoint_path.unlink(missing_ok=True)

    return RunDeletionResult(
        run_id=run_id,
        deleted_event_count=event_count,
        deleted_artifact_count=len(artifact_ids),
    )


@dataclass(frozen=True)
class NodeSummaryLookup:
    node_id: str
    agent_name: str
    summary_artifact_id: str
    summary: NodeSummaryResult


@dataclass(frozen=True)
class NodeReportLookup:
    node_id: str
    report_artifact_id: str
    created_at: str
    report: ReportResult


@dataclass(frozen=True)
class StoredReportLookup:
    run_id: str
    report_artifact_id: str
    created_at: str
    report: ReportResult


def _completed_node_event(*, services: BackendServices, run_id: str, node_id: str):
    completed_event = next(
        (
            event
            for event in reversed(services.run_service.list_events(run_id))
            if event.event_type == "agent.completed"
            and event.metadata.get("node_id") == node_id
        ),
        None,
    )
    if completed_event is None:
        raise NodeSummaryNotFoundError("완료된 노드를 찾을 수 없습니다.")
    return completed_event


def _summary_source_ids(completed_event, summary_metadata: dict[str, Any]) -> list[str]:
    raw_ids = completed_event.artifact_ids or summary_metadata.get("artifact_ids") or []
    return [item for item in raw_ids if isinstance(item, str) and item]


def _latest_summary_artifact_id(
    *, services: BackendServices, run_id: str, source_ids: list[str]
) -> str | None:
    wanted = set(source_ids)
    if not wanted:
        return None
    latest_id: str | None = None
    for artifact in services.artifact_registry.list_artifacts(run_id=run_id, type="file"):
        metadata = artifact.metadata
        if metadata.get("kind") != "node_summary":
            continue
        if set(metadata.get("source_artifact_ids") or []) == wanted:
            latest_id = artifact.artifact_id
    return latest_id


def get_node_summary(
    *,
    services: BackendServices,
    run_id: str,
    node_id: str,
) -> NodeSummaryLookup:
    completed_event = _completed_node_event(services=services, run_id=run_id, node_id=node_id)

    summary_metadata = completed_event.metadata.get("summary")
    summary_metadata = summary_metadata if isinstance(summary_metadata, dict) else {}
    source_ids = _summary_source_ids(completed_event, summary_metadata)
    summary_artifact_id = _latest_summary_artifact_id(
        services=services, run_id=run_id, source_ids=source_ids
    ) or summary_metadata.get("summary_artifact_id")

    if not isinstance(summary_artifact_id, str) or not summary_artifact_id:
        raise NodeSummaryNotFoundError("노드의 상세 서머리가 아직 생성되지 않았습니다.")

    try:
        artifact = services.artifact_registry.get_artifact(summary_artifact_id)
    except Exception as exc:
        raise NodeSummaryNotFoundError("노드의 상세 서머리 artifact를 찾을 수 없습니다.") from exc
    if artifact.run_id != run_id or artifact.metadata.get("kind") != "node_summary":
        raise NodeSummaryNotFoundError("실행에 속한 노드 서머리가 아닙니다.")

    try:
        payload = json.loads(services.artifact_store.read_text(summary_artifact_id))
        summary = NodeSummaryResult.model_validate(payload)
    except Exception as exc:
        raise NodeSummaryNotFoundError("노드의 상세 서머리 형식이 올바르지 않습니다.") from exc

    return NodeSummaryLookup(
        node_id=node_id,
        agent_name=completed_event.node_name or str(summary_metadata.get("agent") or ""),
        summary_artifact_id=summary_artifact_id,
        summary=summary,
    )


def regenerate_node_summary(
    *,
    services: BackendServices,
    run_id: str,
    node_id: str,
) -> NodeSummaryLookup:
    """완료된 노드의 원본 artifact를 유지하고 summary artifact만 새로 만든다."""
    completed_event = _completed_node_event(services=services, run_id=run_id, node_id=node_id)
    summary_metadata = completed_event.metadata.get("summary")
    summary_metadata = summary_metadata if isinstance(summary_metadata, dict) else {}
    source_ids = _summary_source_ids(completed_event, summary_metadata)
    if not source_ids:
        raise NodeSummaryNotFoundError("서머리를 재생성할 원본 artifact가 없습니다.")

    runtime = AgentRuntime(adapter=BackendAdapter(services=services))
    generate_node_summary(source_ids, runtime, force_regenerate=True)
    return get_node_summary(services=services, run_id=run_id, node_id=node_id)


def _report_source_artifact_ids(
    *,
    services: BackendServices,
    run_id: str,
    completed_event,
) -> list[str]:
    summary_metadata = completed_event.metadata.get("summary")
    summary_metadata = summary_metadata if isinstance(summary_metadata, dict) else {}
    direct_ids = _summary_source_ids(completed_event, summary_metadata)
    if not direct_ids:
        raise NodeReportGenerationError("리포트를 생성할 Insight artifact가 없습니다.")

    records = []
    for artifact_id in direct_ids:
        try:
            records.append(services.artifact_registry.get_artifact(artifact_id))
        except BackendError:
            continue

    anchor = next(
        (
            record
            for record in records
            if record.run_id == run_id and record.metadata.get("kind") == "insight_payload"
        ),
        next((record for record in records if record.run_id == run_id), None),
    )
    if anchor is None:
        raise NodeReportGenerationError("선택한 Insight 실행에 속한 artifact를 찾을 수 없습니다.")

    ordered_ids = [anchor.artifact_id]
    seen = set(ordered_ids)
    pending = [
        artifact_id
        for record in records
        for artifact_id in record.parent_ids
    ]
    pending.extend(
        artifact_id for artifact_id in direct_ids if artifact_id != anchor.artifact_id
    )

    while pending:
        artifact_id = pending.pop(0)
        if artifact_id in seen:
            continue
        try:
            artifact = services.artifact_registry.get_artifact(artifact_id)
        except BackendError:
            continue
        seen.add(artifact_id)
        ordered_ids.append(artifact_id)
        pending.extend(artifact.parent_ids)

    return ordered_ids


def generate_node_report(
    *,
    services: BackendServices,
    run_id: str,
    node_id: str,
) -> NodeReportLookup:
    completed_event = _completed_node_event(services=services, run_id=run_id, node_id=node_id)
    agent_name = str(completed_event.metadata.get("agent_name") or completed_event.node_name or "")
    if agent_name != "insight":
        raise NodeReportGenerationError("완료된 Insight 노드에서만 리포트를 생성할 수 있습니다.")

    source_ids = _report_source_artifact_ids(
        services=services,
        run_id=run_id,
        completed_event=completed_event,
    )
    runtime = AgentRuntime(adapter=BackendAdapter(services=services))
    try:
        report_ref = generate_report(source_ids, runtime)
        artifact = services.artifact_registry.get_artifact(report_ref.artifact_id)
        payload = json.loads(services.artifact_store.read_text(report_ref.artifact_id))
        report = ReportResult.model_validate(payload)
    except NodeReportGenerationError:
        raise
    except Exception as exc:
        raise NodeReportGenerationError("리포트 생성 결과를 저장하거나 읽지 못했습니다.") from exc

    if artifact.run_id != run_id or artifact.metadata.get("kind") != "report":
        raise NodeReportGenerationError("선택한 실행에 속한 리포트 artifact가 아닙니다.")

    return NodeReportLookup(
        node_id=node_id,
        report_artifact_id=artifact.artifact_id,
        created_at=artifact.created_at,
        report=report,
    )


def list_session_events(
    *,
    services: BackendServices,
    session_id: str,
) -> list[RunEvent]:
    """세션에 속한 모든 run(메인 쿼리+그 분기 전부)의 이벤트를 시간순으로 모아 반환한다.

    related-events는 run 하나의 branch root 계보만 돌려주지만, 플레이그라운드 캔버스는
    새로고침 후에도 세션에서 시작한 모든 메인 쿼리 트리를 한 번에 그려야 해서 별도로 둔다.
    """
    events: list[RunEvent] = []
    for run in services.run_service.list_runs(project_id=session_id):
        events.extend(services.run_service.list_events(run.run_id))
    events.sort(key=lambda event: (event.created_at or "", event.event_id))
    return events


def list_session_reports(
    *,
    services: BackendServices,
    session_id: str,
) -> list[StoredReportLookup]:
    reports: list[StoredReportLookup] = []
    runs = services.run_service.list_runs(project_id=session_id)
    for run in runs:
        for artifact in services.artifact_registry.list_artifacts(run_id=run.run_id, type="file"):
            if artifact.metadata.get("kind") != "report":
                continue
            try:
                payload = json.loads(services.artifact_store.read_text(artifact.artifact_id))
                report = ReportResult.model_validate(payload)
            except Exception:
                continue
            reports.append(
                StoredReportLookup(
                    run_id=run.run_id,
                    report_artifact_id=artifact.artifact_id,
                    created_at=artifact.created_at,
                    report=report,
                )
            )
    return sorted(reports, key=lambda item: item.created_at, reverse=True)


def new_thread_id() -> str:
    return f"thread_{uuid4().hex}"


class SessionBoundBackendAdapter(BackendAdapter):
    def __init__(
        self,
        *,
        services: BackendServices,
        session: SessionResponse,
        catalog_summary: dict[str, object],
    ) -> None:
        super().__init__(services=services)
        self._session = session
        self._catalog_summary = catalog_summary

    def get_default_datasource_id(self) -> str | None:
        return self._session.id

    def get_catalog_summary(self, datasource_id: str) -> dict | None:
        if datasource_id != self._session.id:
            return None
        return self._catalog_summary


def build_session_catalog_summary(session: SessionResponse) -> dict[str, object]:
    from DATA_Analyst_Assistant_Agent.shared.db import get_db_engine

    with bind_session_database(session):
        engine = get_db_engine()
        if engine is None:
            raise RuntimeError("세션 데이터베이스 연결에 실패했습니다.")
        inspector = inspect(engine)
        summary: dict[str, object] = {}
        for table in inspector.get_table_names():
            summary[table] = {
                "primary_key": inspector.get_pk_constraint(table).get("constrained_columns", []),
                "description": "",
                "columns": [
                    {
                        "name": column["name"],
                        "type": str(column["type"]),
                        "nullable": bool(column["nullable"]),
                        "description": "",
                    }
                    for column in inspector.get_columns(table)
                ],
                "foreign_keys": [
                    {
                        "referred_table": fk.get("referred_table"),
                        "referred_columns": fk.get("referred_columns", []),
                        "constrained_columns": fk.get("constrained_columns", []),
                    }
                    for fk in inspector.get_foreign_keys(table)
                ],
            }
        engine.dispose()
        return summary


@contextmanager
def bind_session_database(session: SessionResponse) -> Iterator[None]:
    from DATA_Analyst_Assistant_Agent.agents.sql._runtime import reset_engine_cache

    with _SESSION_ENV_LOCK:
        keys = {
            "MYSQL_HOST": StorageMySQL.HOST,
            "MYSQL_PORT": str(StorageMySQL.PORT),
            "MYSQL_USERNAME": StorageMySQL.USER,
            "MYSQL_PASSWORD": StorageMySQL.PASSWORD,
            "MYSQL_DATABASE": session.session_db,
            "DB_HOST": StorageMySQL.HOST,
            "DB_PORT": str(StorageMySQL.PORT),
            "DB_USER": StorageMySQL.USER,
            "DB_PASSWORD": StorageMySQL.PASSWORD,
            "DB_NAME": session.session_db,
        }
        previous = {key: os.environ.get(key) for key in keys}
        try:
            os.environ.update(keys)
            reset_engine_cache()
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            reset_engine_cache()


def launch_agent_run(
    *,
    services: BackendServices,
    session: SessionResponse,
    username: str,
    query: str,
    run_id: str,
    thread_id: str,
) -> None:
    try:
        try:
            services.run_service.claim_created(
                run_id,
                metadata={
                    "session_id": session.id,
                    "session_db": session.session_db,
                    "owner": username,
                },
            )
        except BackendError as exc:
            current_status = services.run_service.get_run(run_id).status
            if exc.code == "RUN_NOT_CREATED" and current_status == RunStatus.cancelled:
                return
            raise

        event_adapter = BackendAdapter(services=services)
        event_adapter.append_run_event(
            run_id,
            "run.started",
            "Agent workflow started.",
            node_name="supervisor",
            metadata={"session_id": session.id, "thread_id": thread_id},
        )
        catalog_summary = build_session_catalog_summary(session)
        adapter = SessionBoundBackendAdapter(
            services=services,
            session=session,
            catalog_summary=catalog_summary,
        )
        raise_if_run_cancelled(adapter, run_id)
        supervisor = SupervisorAgent(adapter, checkpoint_path=adapter.base_data_dir / f"{thread_id}.sqlite")
        initial_state = empty_supervisor_state(
            thread_id=thread_id,
            run_id=run_id,
            user_query=query,
            datasource_id=session.id,
            project_id=session.id,
            catalog_summary=catalog_summary,
        )
        with bind_session_database(session):
            output = supervisor._invoke_graph(initial_state, thread_id)
        interrupt = supervisor._interrupt_payload_from_output(output)
        if interrupt is not None:
            supervisor._update_run_from_interrupt(run_id, interrupt)
            return
        result = supervisor._update_run_from_terminal_output(run_id, output)
        _append_terminal_event(services, run_id, result)
    except RunCancellationRequested:
        return
    except Exception as exc:
        _mark_run_failed(services, run_id, session.id, exc)
        raise


_RESUME_MESSAGES = {
    "clarification": "Clarification answer received. Resuming agent workflow.",
    "analysis_review": "Analysis review decision received. Resuming agent workflow.",
    "approval": "Approval received. Resuming agent workflow.",
}


def resume_agent_run(
    *,
    services: BackendServices,
    session: SessionResponse,
    resume_payload: dict[str, Any],
    run_id: str,
    thread_id: str,
) -> None:
    try:
        run = services.run_service.get_run(run_id)
        interrupt_node = run.metadata.get("node")
        node_name = interrupt_node if isinstance(interrupt_node, str) and interrupt_node else "supervisor"
        resumed_from = str(run.metadata.get("resumed_from") or "clarification")
        event_adapter = BackendAdapter(services=services)
        event_adapter.append_run_event(
            run_id,
            "human_input.resumed",
            _RESUME_MESSAGES.get(resumed_from, _RESUME_MESSAGES["clarification"]),
            node_name=node_name,
            metadata={
                "interrupt_type": resumed_from,
                "thread_id": thread_id,
                "node": node_name,
            },
        )
        catalog_summary = build_session_catalog_summary(session)
        adapter = SessionBoundBackendAdapter(
            services=services,
            session=session,
            catalog_summary=catalog_summary,
        )
        raise_if_run_cancelled(adapter, run_id)
        supervisor = SupervisorAgent(adapter, checkpoint_path=adapter.base_data_dir / f"{thread_id}.sqlite")
        with bind_session_database(session):
            result = supervisor.resume(thread_id, resume_payload)
        if isinstance(result, SupervisorRunResult) and result.kind == "state":
            _append_terminal_event(services, run_id, result)
    except RunCancellationRequested:
        return
    except Exception as exc:
        _mark_run_failed(services, run_id, session.id, exc)
        raise


_BRANCH_STAGE_ORDER: list[BranchStage] = ["sql", "eda", "analysis", "insight"]
_BRANCH_STAGE_AGENT = {
    "sql": "sql_agent",
    "eda": "eda_agent",
    "analysis": "analysis_agent",
    "insight": "insight",
}
_BRANCH_AGENT_STAGE = {agent: stage for stage, agent in _BRANCH_STAGE_AGENT.items()}


class BranchPlanError(Exception):
    """분기를 시작하기 전 사전 검증에서 실패했을 때(체크포인트 없음/선행 단계 결과 없음 등)."""


@dataclass
class BranchPlan:
    upstream_artifact_ids: dict[str, list[str]]
    original_question: str
    target_table: str | None
    default_parent_node_id: str | None


def _run_lineage(*, services: BackendServices, run: RunRecord) -> list[RunRecord]:
    lineage = [run]
    seen = {run.run_id}
    current = run
    while True:
        parent_run_id = current.metadata.get("branched_from_run_id")
        if not isinstance(parent_run_id, str) or not parent_run_id:
            break
        if parent_run_id in seen:
            raise BranchPlanError("분기 실행 계보에 순환 참조가 있습니다.")
        try:
            parent = services.run_service.get_run(parent_run_id)
        except Exception as exc:
            raise BranchPlanError("상위 분기 실행을 찾을 수 없습니다.") from exc
        seen.add(parent.run_id)
        lineage.append(parent)
        current = parent
    return list(reversed(lineage))


def _event_artifact_ids(event) -> list[str]:
    ids = event.artifact_ids
    if not ids:
        summary = event.metadata.get("summary")
        if isinstance(summary, dict):
            ids = summary.get("artifact_ids") or []
    return [artifact_id for artifact_id in ids if isinstance(artifact_id, str) and artifact_id]


def _effective_branch_stage_outputs(
    *,
    services: BackendServices,
    lineage: list[RunRecord],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    artifact_ids_by_agent: dict[str, list[str]] = {}
    node_id_by_agent: dict[str, str] = {}

    for lineage_run in lineage:
        for event in services.run_service.list_events(lineage_run.run_id):
            if event.event_type != "agent.completed" or event.node_name not in _BRANCH_AGENT_STAGE:
                continue
            stage_artifact_ids = _event_artifact_ids(event)
            if stage_artifact_ids:
                artifact_ids_by_agent[event.node_name] = stage_artifact_ids
            node_id = event.metadata.get("node_id")
            if isinstance(node_id, str) and node_id:
                node_id_by_agent[event.node_name] = node_id

    return artifact_ids_by_agent, node_id_by_agent


def _default_branch_parent_node_id(
    *,
    services: BackendServices,
    run_id: str,
    start_stage: BranchStage,
    checkpoint_state: dict[str, Any],
    node_id_by_agent: dict[str, str] | None = None,
) -> str | None:
    """분기 시작 노드는 start_stage 직전 완료 노드에 붙인다.

    예: EDA부터 다시 돌면 원본 SQL 노드 뒤에 새 EDA가 붙어야 하므로, 마지막 완료 노드
    (보통 Insight)가 아니라 직전 단계(SQL)의 completed node_id를 사용한다.
    """
    start_index = _BRANCH_STAGE_ORDER.index(start_stage)
    if start_index <= 0:
        return None

    previous_stage = _BRANCH_STAGE_ORDER[start_index - 1]
    previous_agent = _BRANCH_STAGE_AGENT[previous_stage]
    if node_id_by_agent and node_id_by_agent.get(previous_agent):
        return node_id_by_agent[previous_agent]

    latest_node_id: str | None = None

    for event in services.run_service.list_events(run_id):
        if event.event_type != "agent.completed" or event.node_name != previous_agent:
            continue
        node_id = event.metadata.get("node_id")
        if isinstance(node_id, str) and node_id:
            latest_node_id = node_id

    if latest_node_id:
        return latest_node_id

    fallback = checkpoint_state.get("last_completed_node_id")
    return fallback if isinstance(fallback, str) else None


def prepare_branch_plan(
    *,
    services: BackendServices,
    run: RunRecord,
    start_stage: BranchStage,
) -> BranchPlan:
    """run의 체크포인트에서 start_stage 이전 단계까지의 artifact_ids/질문/target_table을 읽어온다."""
    if not run.thread_id:
        raise BranchPlanError("실행에 연결된 thread 정보가 없습니다.")

    adapter = BackendAdapter(services=services)
    supervisor = SupervisorAgent(adapter, checkpoint_path=adapter.base_data_dir / f"{run.thread_id}.sqlite")
    checkpoint_state = supervisor.get_checkpoint_state(run.thread_id)
    if not checkpoint_state:
        raise BranchPlanError("실행 체크포인트를 찾을 수 없습니다.")

    orchestration_state = to_orchestration_state(checkpoint_state)
    start_index = _BRANCH_STAGE_ORDER.index(start_stage)
    upstream_stages = _BRANCH_STAGE_ORDER[:start_index]
    lineage = _run_lineage(services=services, run=run)
    effective_artifact_ids, effective_node_ids = _effective_branch_stage_outputs(
        services=services,
        lineage=lineage,
    )
    checkpoint_artifact_ids = dict(orchestration_state.artifact_ids)
    checkpoint_artifact_ids.update(effective_artifact_ids)
    upstream_artifact_ids = {
        _BRANCH_STAGE_AGENT[stage]: checkpoint_artifact_ids.get(_BRANCH_STAGE_AGENT[stage], [])
        for stage in upstream_stages
        if checkpoint_artifact_ids.get(_BRANCH_STAGE_AGENT[stage])
    }
    missing_stages = [
        stage for stage in upstream_stages if not upstream_artifact_ids.get(_BRANCH_STAGE_AGENT[stage])
    ]
    if missing_stages:
        raise BranchPlanError(f"{', '.join(missing_stages)} 단계 결과가 없어 분기를 시작할 수 없습니다.")

    default_parent_node_id = _default_branch_parent_node_id(
        services=services,
        run_id=run.run_id,
        start_stage=start_stage,
        checkpoint_state=checkpoint_state,
        node_id_by_agent=effective_node_ids,
    )
    return BranchPlan(
        upstream_artifact_ids=upstream_artifact_ids,
        original_question=orchestration_state.user_query,
        target_table=orchestration_state.plan.target_table if orchestration_state.plan else None,
        default_parent_node_id=default_parent_node_id,
    )


def run_branch_task(
    *,
    services: BackendServices,
    session: SessionResponse,
    run_id: str,
    thread_id: str,
    start_stage: BranchStage,
    instruction: str,
    upstream_artifact_ids: dict[str, list[str]],
    original_question: str,
    target_table: str | None,
    parent_node_id: str | None,
) -> None:
    try:
        try:
            services.run_service.claim_created(
                run_id,
                metadata={"branch_stage": start_stage, "branch_instruction": instruction},
            )
        except BackendError as exc:
            current_status = services.run_service.get_run(run_id).status
            if exc.code == "RUN_NOT_CREATED" and current_status == RunStatus.cancelled:
                return
            raise

        event_adapter = BackendAdapter(services=services)
        event_adapter.append_run_event(
            run_id,
            "branch.started",
            f"'{instruction}' 지시사항으로 {start_stage} 단계부터 분기를 시작합니다.",
            node_name="supervisor",
            metadata={"branch": True, "start_stage": start_stage},
        )
        catalog_summary = build_session_catalog_summary(session)
        adapter = SessionBoundBackendAdapter(
            services=services,
            session=session,
            catalog_summary=catalog_summary,
        )
        raise_if_run_cancelled(adapter, run_id)
        runtime = AgentRuntime(adapter)
        with bind_session_database(session):
            result = branch_from(
                start_stage,
                instruction,
                upstream_artifact_ids=upstream_artifact_ids,
                original_question=original_question,
                run_id=run_id,
                thread_id=thread_id,
                runtime=runtime,
                backend_adapter=adapter,
                parent_node_id=parent_node_id,
                target_table=target_table,
            )
    except RunCancellationRequested:
        return
    except Exception as exc:
        _mark_run_failed(services, run_id, session.id, exc)
        raise

    if result.failed_agent:
        services.run_service.update_status(
            run_id,
            RunStatus.failed,
            metadata={
                "error": result.failure_reason,
                "session_id": session.id,
                "branch_failed_agent": result.failed_agent,
            },
        )
        services.run_service.append_event(
            run_id,
            "run.failed",
            f"{result.failed_agent} 분기 실행 중 오류가 발생했습니다: {result.failure_reason}",
            node_name="supervisor",
            metadata={"session_id": session.id, "branch": True},
        )
        return

    services.run_service.update_status(
        run_id,
        RunStatus.succeeded,
        metadata={"session_id": session.id, "branch_completed": True},
    )
    services.run_service.append_event(
        run_id,
        "run.completed",
        "분기 실행이 완료됐습니다.",
        node_name="supervisor",
        metadata={"session_id": session.id, "branch": True, "new_artifact_ids": result.artifact_ids},
    )


def _append_terminal_event(
    services: BackendServices,
    run_id: str,
    result: SupervisorRunResult,
) -> None:
    state = result.state
    if state is None or state.terminal_state is None:
        return
    if state.terminal_state == SupervisorTerminalState.needs_user_approval:
        return

    completed = state.terminal_state == SupervisorTerminalState.completed
    services.run_service.append_event(
        run_id,
        "run.completed" if completed else "run.failed",
        state.final_answer or ("Agent workflow finished." if completed else "Agent workflow failed."),
        node_name="supervisor",
        metadata={"terminal_state": state.terminal_state.value},
    )


def _mark_run_failed(
    services: BackendServices,
    run_id: str,
    session_id: str,
    exc: Exception,
) -> None:
    run = services.run_service.get_run(run_id)
    if run.status == RunStatus.cancelled:
        return
    if run.status != RunStatus.failed:
        try:
            services.run_service.update_status(
                run_id,
                RunStatus.failed,
                metadata={"error": str(exc), "session_id": session_id},
            )
        except BackendError as update_exc:
            current_status = services.run_service.get_run(run_id).status
            if update_exc.code == "RUN_TERMINAL" and current_status == RunStatus.cancelled:
                return
            raise
    if services.run_service.get_run(run_id).status == RunStatus.cancelled:
        return
    services.run_service.append_event(
        run_id,
        "run.failed",
        str(exc),
        node_name="supervisor",
        metadata={"session_id": session_id},
    )

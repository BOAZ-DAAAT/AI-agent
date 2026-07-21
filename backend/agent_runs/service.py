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
from data_agent_backend.models.runs import TERMINAL_RUN_STATUSES, RunRecord, RunStatus
from data_agent_backend.services.factory import BackendServices
from data_agent_backend.storage.filesystem import ensure_child_path

from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.shared.contracts import SupervisorRunResult, SupervisorTerminalState
from DATA_Analyst_Assistant_Agent.supervisor.agent import SupervisorAgent
from DATA_Analyst_Assistant_Agent.supervisor.branch import BranchStage, branch_from
from DATA_Analyst_Assistant_Agent.supervisor.state import empty_supervisor_state, to_orchestration_state
from DATA_Analyst_Assistant_Agent.supervisor.summary.generator import generate_node_summary
from DATA_Analyst_Assistant_Agent.supervisor.summary.schemas import NodeSummaryResult


_SESSION_ENV_LOCK = threading.Lock()


class NodeSummaryNotFoundError(Exception):
    """완료 노드 또는 해당 노드의 상세 서머리를 찾지 못했을 때."""


class RunDeletionConflictError(Exception):
    """아직 실행 중인 run 삭제를 요청했을 때."""


@dataclass(frozen=True)
class RunDeletionResult:
    run_id: str
    deleted_event_count: int
    deleted_artifact_count: int


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
    services.run_service.update_status(
        run_id,
        RunStatus.running,
        metadata={"session_id": session.id, "session_db": session.session_db, "owner": username},
    )
    services.run_service.append_event(
        run_id,
        "run.started",
        "Agent workflow started.",
        node_name="supervisor",
        metadata={"session_id": session.id, "thread_id": thread_id},
    )

    try:
        catalog_summary = build_session_catalog_summary(session)
        adapter = SessionBoundBackendAdapter(services=services, session=session, catalog_summary=catalog_summary)
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
    run = services.run_service.get_run(run_id)
    interrupt_node = run.metadata.get("node")
    node_name = interrupt_node if isinstance(interrupt_node, str) and interrupt_node else "supervisor"
    resumed_from = str(run.metadata.get("resumed_from") or "clarification")
    services.run_service.append_event(
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

    try:
        catalog_summary = build_session_catalog_summary(session)
        adapter = SessionBoundBackendAdapter(services=services, session=session, catalog_summary=catalog_summary)
        supervisor = SupervisorAgent(adapter, checkpoint_path=adapter.base_data_dir / f"{thread_id}.sqlite")
        with bind_session_database(session):
            result = supervisor.resume(thread_id, resume_payload)
        if isinstance(result, SupervisorRunResult) and result.kind == "state":
            _append_terminal_event(services, run_id, result)
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


class BranchPlanError(Exception):
    """분기를 시작하기 전 사전 검증에서 실패했을 때(체크포인트 없음/선행 단계 결과 없음 등)."""


@dataclass
class BranchPlan:
    upstream_artifact_ids: dict[str, list[str]]
    original_question: str
    target_table: str | None
    default_parent_node_id: str | None


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
    upstream_artifact_ids = {
        _BRANCH_STAGE_AGENT[stage]: orchestration_state.artifact_ids.get(_BRANCH_STAGE_AGENT[stage], [])
        for stage in upstream_stages
        if orchestration_state.artifact_ids.get(_BRANCH_STAGE_AGENT[stage])
    }
    missing_stages = [
        stage for stage in upstream_stages if not upstream_artifact_ids.get(_BRANCH_STAGE_AGENT[stage])
    ]
    if missing_stages:
        raise BranchPlanError(f"{', '.join(missing_stages)} 단계 결과가 없어 분기를 시작할 수 없습니다.")

    default_parent_node_id = checkpoint_state.get("last_completed_node_id")
    return BranchPlan(
        upstream_artifact_ids=upstream_artifact_ids,
        original_question=orchestration_state.user_query,
        target_table=orchestration_state.plan.target_table if orchestration_state.plan else None,
        default_parent_node_id=default_parent_node_id if isinstance(default_parent_node_id, str) else None,
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
    services.run_service.update_status(
        run_id,
        RunStatus.running,
        metadata={"branch_stage": start_stage, "branch_instruction": instruction},
    )
    services.run_service.append_event(
        run_id,
        "branch.started",
        f"'{instruction}' 지시사항으로 {start_stage} 단계부터 분기를 시작합니다.",
        node_name="supervisor",
        metadata={"branch": True, "start_stage": start_stage},
    )

    try:
        catalog_summary = build_session_catalog_summary(session)
        adapter = SessionBoundBackendAdapter(services=services, session=session, catalog_summary=catalog_summary)
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
    try:
        run = services.run_service.get_run(run_id)
        if run.status != RunStatus.failed:
            services.run_service.update_status(
                run_id,
                RunStatus.failed,
                metadata={"error": str(exc), "session_id": session_id},
            )
    finally:
        services.run_service.append_event(
            run_id,
            "run.failed",
            str(exc),
            node_name="supervisor",
            metadata={"session_id": session_id},
        )

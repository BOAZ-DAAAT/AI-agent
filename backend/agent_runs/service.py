from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Iterator
from uuid import uuid4

from sqlalchemy import inspect

from backend.config import StorageMySQL
from backend.session.schemas import SessionResponse
from data_agent_backend.models.runs import RunStatus
from data_agent_backend.services.factory import BackendServices

from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.shared.contracts import SupervisorRunResult, SupervisorTerminalState
from DATA_Analyst_Assistant_Agent.supervisor.agent import SupervisorAgent
from DATA_Analyst_Assistant_Agent.supervisor.state import empty_supervisor_state


_SESSION_ENV_LOCK = threading.Lock()


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


def resume_agent_run(
    *,
    services: BackendServices,
    session: SessionResponse,
    answer: str,
    run_id: str,
    thread_id: str,
) -> None:
    run = services.run_service.get_run(run_id)
    interrupt_node = run.metadata.get("node")
    node_name = interrupt_node if isinstance(interrupt_node, str) and interrupt_node else "supervisor"
    services.run_service.append_event(
        run_id,
        "human_input.resumed",
        "Clarification answer received. Resuming agent workflow.",
        node_name=node_name,
        metadata={
            "interrupt_type": "clarification",
            "thread_id": thread_id,
            "node": node_name,
        },
    )

    try:
        catalog_summary = build_session_catalog_summary(session)
        adapter = SessionBoundBackendAdapter(services=services, session=session, catalog_summary=catalog_summary)
        supervisor = SupervisorAgent(adapter, checkpoint_path=adapter.base_data_dir / f"{thread_id}.sqlite")
        with bind_session_database(session):
            result = supervisor.resume(thread_id, {"answer": answer})
        if isinstance(result, SupervisorRunResult) and result.kind == "state":
            _append_terminal_event(services, run_id, result)
    except Exception as exc:
        _mark_run_failed(services, run_id, session.id, exc)
        raise


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

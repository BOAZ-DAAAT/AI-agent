from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from data_agent_backend.models.runs import RunStatus
from langgraph.types import Command

from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState, SupervisorTerminalState
from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model
from DATA_Analyst_Assistant_Agent.supervisor.checkpoint import open_sqlite_checkpointer
from DATA_Analyst_Assistant_Agent.supervisor.graph import build_graph
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    SupervisorState,
    empty_supervisor_state,
    to_orchestration_state,
)
from DATA_Analyst_Assistant_Agent.supervisor.tools import SubAgentAdapter


class SupervisorAgent:
    def __init__(
        self,
        adapter: Any | None = None,
        *,
        checkpoint_path: str | Path | None = None,
        model: Any | None = None,
    ) -> None:
        self.adapter = adapter or BackendAdapter()
        self.checkpoint_path = checkpoint_path
        self.model = model

    def run(
        self,
        query: str,
        *,
        thread_id: str | None = None,
        datasource_id: str | None = None,
        project_id: str | None = None,
    ) -> OrchestrationState:
        thread_id = thread_id or f"thread_{uuid4().hex}"
        run = self.adapter.create_run(
            thread_id=thread_id,
            project_id=project_id,
            metadata={
                "query": query,
                "supervisor": "langgraph",
            },
        )

        try:
            datasource_id = self._resolve_datasource_id(datasource_id)
            catalog_summary = self._resolve_catalog_summary(datasource_id)
            initial_state = empty_supervisor_state(
                thread_id=thread_id,
                run_id=run.run_id,
                user_query=query,
                datasource_id=datasource_id,
                project_id=project_id,
                catalog_summary=catalog_summary,
            )
            output = self._invoke_graph(initial_state, thread_id)
            return self._update_run_from_terminal_output(run.run_id, output)
        except Exception as exc:
            self.adapter.update_run_status(run.run_id, RunStatus.failed, metadata={"error": str(exc)})
            raise

    def resume(self, thread_id: str, resume_payload: dict[str, Any]) -> Any:
        config = {"configurable": {"thread_id": thread_id}}
        with open_sqlite_checkpointer(self.checkpoint_path) as checkpointer:
            graph = self._build_runtime_graph(checkpointer)
            result = self._resume_synthetic_approval(graph, config, resume_payload)
            if result is None:
                result = graph.invoke(Command(resume=resume_payload), config)
        return self._update_run_status_from_resume_result(result)

    def _resume_synthetic_approval(
        self,
        graph: Any,
        config: dict[str, Any],
        resume_payload: dict[str, Any],
    ) -> Any | None:
        if resume_payload.get("approved") is not True:
            return None
        if not hasattr(graph, "get_state") or not hasattr(graph, "update_state"):
            return None

        latest_state = graph.get_state(config)
        values = getattr(latest_state, "values", None)
        if not isinstance(values, dict):
            return None

        pending_approval = values.get("pending_approval")
        if not isinstance(pending_approval, dict):
            return None

        next_action = self._next_action_for_pending_agent(pending_approval.get("agent"))
        if next_action is None:
            return None

        updates = {
            "pending_approval": None,
            "terminal_state": "running",
            "next_action": next_action,
            "final_answer": "",
        }
        returned_config = graph.update_state(config, updates, as_node="summarize_step")
        return graph.invoke(None, returned_config or config)

    @staticmethod
    def _next_action_for_pending_agent(agent_name: Any) -> str | None:
        return {
            "sql_agent": "call_sql_agent",
            "eda_agent": "call_eda_agent",
            "analysis_agent": "call_analysis_agent",
            "report_agent": "call_report_agent",
        }.get(str(agent_name))

    def _update_run_status_from_resume_result(self, result: Any) -> Any:
        if not isinstance(result, dict):
            return result

        run_id = result.get("current_run_id")
        if not isinstance(run_id, str) or not run_id:
            return result

        terminal_state = result.get("terminal_state")
        if not terminal_state or terminal_state == "running":
            return result

        try:
            self._update_run_from_terminal_output(run_id, result)
        except Exception as exc:
            self.adapter.update_run_status(run_id, RunStatus.failed, metadata={"error": str(exc)})
            raise
        return result

    def _invoke_graph(self, initial_state: SupervisorState, thread_id: str) -> SupervisorState:
        with open_sqlite_checkpointer(self.checkpoint_path) as checkpointer:
            graph = self._build_runtime_graph(checkpointer)
            return graph.invoke(initial_state, {"configurable": {"thread_id": thread_id}})

    def _build_runtime_graph(self, checkpointer: Any):
        return build_graph(
            SubAgentAdapter(backend_adapter=self.adapter),
            model=self._decision_model(),
            checkpointer=checkpointer,
        )

    def _decision_model(self) -> Any:
        if self.model is not None:
            return self.model
        return get_chat_model(temperature=0)

    def _resolve_datasource_id(self, datasource_id: str | None) -> str | None:
        # Datasource is optional for the supervisor flow.
        # Keep explicit datasource support, but do not auto-resolve a default
        # datasource when the caller did not request one.
        return datasource_id

    def _resolve_catalog_summary(self, datasource_id: str | None) -> dict[str, Any] | None:
        if datasource_id is None:
            return None
        getter = getattr(self.adapter, "get_catalog_summary", None)
        if getter is None:
            return None
        return getter(datasource_id)

    def _update_run_from_terminal_output(self, run_id: str, output: SupervisorState) -> OrchestrationState:
        orchestration_state = to_orchestration_state(output)
        self.adapter.update_run_status(
            run_id,
            self._run_status_for_terminal(orchestration_state.terminal_state),
            metadata={"terminal_state": output.get("terminal_state")},
        )
        return orchestration_state

    @staticmethod
    def _run_status_for_terminal(terminal_state: SupervisorTerminalState | None) -> RunStatus:
        if terminal_state == SupervisorTerminalState.completed:
            return RunStatus.succeeded
        if terminal_state == SupervisorTerminalState.needs_user_approval:
            return RunStatus.waiting_approval
        # 백엔드에는 clarification 전용 상태가 없으므로 나머지 terminal 상태는 실패로 기록한다.
        return RunStatus.failed


SQLAgentSupervisor = SupervisorAgent

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from data_agent_backend.models.approvals import ApprovalStatus
from data_agent_backend.models.runs import RunStatus

from DATA_Analyst_Assistant_Agent.agents import (
    AgentRuntime,
    AnalysisAgent,
    CentralValidationAgent,
    EDAAgent,
    ReportAgent,
    SQLAgent,
    VisualizationAgent,
)
from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.shared.contracts import (
    AgentEnvelope,
    AgentStatus,
    AnalysisPlan,
    OrchestrationState,
    SupervisorTerminalState,
)

AGENT_LOG_LABELS = {
    "sql_agent": "SQL agent",
    "eda_agent": "EDA agent",
    "analysis_agent": "Analysis agent",
    "visualization_agent": "Visualization agent",
    "report_agent": "Report agent",
    "validation_agent": "Validation agent",
}


class SQLAgentSupervisor:
    """LangGraph-compatible orchestration supervisor for DATA_Analyst_Assistant_Agent."""

    def __init__(
        self,
        adapter: BackendAdapter,
        *,
        sql_agent: SQLAgent | None = None,
        validation_agent: CentralValidationAgent | None = None,
        eda_agent: EDAAgent | None = None,
        analysis_agent: AnalysisAgent | None = None,
        visualization_agent: VisualizationAgent | None = None,
        report_agent: ReportAgent | None = None,
        build_runtime_graph: bool = True,
    ) -> None:
        self.adapter = adapter
        self.runtime = AgentRuntime(adapter)
        self.sql_agent = sql_agent or SQLAgent()
        self.validation_agent = validation_agent or CentralValidationAgent()
        self.eda_agent = eda_agent or EDAAgent()
        self.analysis_agent = analysis_agent or AnalysisAgent()
        self.visualization_agent = visualization_agent or VisualizationAgent()
        self.report_agent = report_agent or ReportAgent()
        self.agent_registry = {
            self.sql_agent.name: self.sql_agent,
            self.eda_agent.name: self.eda_agent,
            self.analysis_agent.name: self.analysis_agent,
            self.visualization_agent.name: self.visualization_agent,
            self.report_agent.name: self.report_agent,
        }
        self.graph = None
        if build_runtime_graph:
            from DATA_Analyst_Assistant_Agent.supervisor.graph import build_graph

            self.graph = build_graph(adapter, supervisor=self)

    def run(
        self,
        user_query: str,
        *,
        thread_id: str | None = None,
        datasource_id: str | None = None,
    ) -> OrchestrationState:
        run = self.adapter.create_run(thread_id=thread_id, metadata={"entrypoint": "sql_agent_supervisor"})
        resolved_datasource_id = datasource_id or self.adapter.get_default_datasource_id()
        state = OrchestrationState(
            run_id=run.run_id,
            thread_id=thread_id,
            user_query=user_query,
            datasource_id=resolved_datasource_id,
        )
        self.adapter.update_run_status(state.run_id, RunStatus.running)
        self._log("Orchestration 실행 시작")
        self.adapter.append_run_event(
            state.run_id,
            "supervisor.started",
            "Supervisor started.",
            node_name="start",
            metadata={"datasource_id": resolved_datasource_id},
        )

        try:
            graph = self.graph
            if graph is None:
                from DATA_Analyst_Assistant_Agent.supervisor.graph import build_graph

                graph = build_graph(self.adapter, supervisor=self)
            result = graph.invoke(
                {
                    "state": state,
                    "last_agent_result": None,
                    "last_validation_result": None,
                    "gate_result": None,
                }
            )
            return result["state"]
        except Exception as exc:
            state.error_state = {"type": type(exc).__name__, "message": str(exc), "step": state.current_step}
            state.terminal_state = SupervisorTerminalState.failed_terminal
            self.adapter.append_run_event(
                state.run_id,
                "supervisor.failed",
                str(exc),
                node_name=state.current_step,
                metadata=state.error_state,
            )
            self.adapter.update_run_status(state.run_id, RunStatus.failed, metadata={"terminal_state": state.terminal_state.value})
            return state

    def parse_plan(self, graph_state: dict[str, Any]) -> dict[str, Any]:
        state = graph_state["state"]
        query = state.user_query
        query_lower = query.lower()
        self._log("Plan agent 실행 시작")

        requires_mart = any(token in query for token in ("반복", "데이터마트", "마트", "저장", "재사용"))
        wants_trend = any(token in query_lower for token in ("trend", "monthly", "chart")) or any(
            token in query for token in ("월별", "추이", "차트", "시각화")
        )
        wants_eda = any(token in query_lower for token in ("eda", "profile", "profiling", "quality")) or any(
            token in query for token in ("프로파일", "품질", "컬럼", "결측")
        )
        wants_analysis = any(token in query_lower for token in ("analysis", "analyze", "insight")) or any(
            token in query for token in ("분석", "인사이트")
        )

        # Demo/user-facing Korean queries should route by intent, not only by the
        # older mojibake keyword set above.
        requires_mart = requires_mart or any(
            token in query_lower
            for token in ("mart", "data mart", "reuse", "persist", "반복", "데이터마트", "마트", "저장", "재사용")
        )
        wants_trend = wants_trend or any(
            token in query_lower
            for token in ("trend", "monthly", "chart", "visualization", "시계열", "월별", "추이", "차트", "시각화")
        )
        wants_eda = wants_eda or any(
            token in query_lower
            for token in (
                "eda",
                "profile",
                "profiling",
                "quality",
                "distribution",
                "outlier",
                "missing",
                "프로파일",
                "품질",
                "분포",
                "이상치",
                "결측",
                "상위",
                "평균",
                "율",
                "비율",
            )
        )
        wants_analysis = wants_analysis or any(
            token in query_lower
            for token in (
                "analysis",
                "analyze",
                "insight",
                "compare",
                "rank",
                "calculate",
                "분석",
                "계산",
                "비교",
                "순위",
                "상위",
                "평균",
                "지연율",
                "리뷰",
                "보여줘",
            )
        )

        if requires_mart:
            route_kind = "mart"
            goal = "재사용 가능한 마트 후보 검토"
            remaining_agents = [self.sql_agent.name]
        elif (wants_eda and wants_trend) or (wants_eda and wants_analysis):
            route_kind = "comprehensive"
            goal = "종합 데이터 분석 (프로파일 + 분석 + 시각화)"
            remaining_agents = [
                self.sql_agent.name, self.eda_agent.name, self.analysis_agent.name,
                self.visualization_agent.name, self.report_agent.name,
            ]
        elif wants_trend:
            route_kind = "trend"
            goal = "월별 추이 분석"
            remaining_agents = [self.sql_agent.name, self.analysis_agent.name, self.visualization_agent.name, self.report_agent.name]
        elif wants_eda:
            route_kind = "eda"
            goal = "데이터 프로파일 및 품질 요약"
            remaining_agents = [self.sql_agent.name, self.eda_agent.name, self.report_agent.name]
        else:
            route_kind = "simple"
            goal = "SQL 기반 질의 응답"
            remaining_agents = [self.sql_agent.name, self.report_agent.name]

        state.goal = goal
        state.route_kind = route_kind
        state.remaining_agents = remaining_agents
        catalog_summary = self._resolve_catalog_summary(state.datasource_id)
        planner_mode = "llm" if state.datasource_id and catalog_summary else "deterministic"
        retry_context = self._retry_context_from_state(state)
        state.catalog_summary = catalog_summary
        state.planner_mode = planner_mode
        state.plan = AnalysisPlan(
            goal=goal,
            metric="매출" if "매출" in query else None,
            dimension="월" if wants_trend else None,
            requires_mart_review=requires_mart,
            route_kind=route_kind,
            datasource_id=state.datasource_id,
            catalog_summary=catalog_summary,
            retry_context=retry_context,
            planner_mode=planner_mode,
            generated_sql="SELECT 1 AS sample_value",
            source_sql="SELECT 1 AS sample_value",
        )
        state.current_step = "parse_plan"
        self._log(f"Plan agent 완료: route={route_kind}, 다음 단계={', '.join(remaining_agents)}")
        self.adapter.append_run_event(
            state.run_id,
            "supervisor.plan_created",
            "Execution plan created.",
            node_name="parse_plan",
            metadata={
                **state.plan.model_dump(mode="json"),
                "remaining_agents": list(state.remaining_agents),
            },
        )
        return {
            **graph_state,
            "state": state,
            "last_agent_result": None,
            "last_validation_result": None,
            "gate_result": None,
        }

    def _resolve_catalog_summary(self, datasource_id: str | None) -> dict[str, Any] | None:
        if not datasource_id:
            return None
        summary = self.adapter.get_catalog_summary(datasource_id)
        if self._catalog_is_empty(summary):
            summary = self.adapter.refresh_catalog(datasource_id)
        return summary

    @staticmethod
    def _catalog_is_empty(summary: dict[str, Any] | None) -> bool:
        if not summary:
            return True
        return not bool(summary.get("tables"))

    @staticmethod
    def _retry_context_from_state(state: OrchestrationState) -> dict[str, Any] | None:
        if not state.error_state:
            return None
        if not any(key in state.error_state for key in ("code", "message", "query", "datasource_id")):
            return None
        return {
            "code": state.error_state.get("code") or state.error_state.get("type") or "unknown_error",
            "message": state.error_state.get("message", ""),
            "query": state.error_state.get("query", ""),
            "datasource_id": state.error_state.get("datasource_id") or state.datasource_id,
            "step": state.error_state.get("step", state.current_step),
        }

    # ── explicit agent nodes ──────────────────────────────────────────────
    #
    # Each node runs one specific specialist agent. Routing remains
    # plan-driven: ``state.remaining_agents`` (built in ``parse_plan``)
    # decides which node executes next; the nodes themselves are fixed,
    # meaningful graph steps rather than a generic dispatcher.

    def _run_agent(self, graph_state: dict[str, Any], agent_name: str) -> dict[str, Any]:
        state = graph_state["state"]
        agent = self.agent_registry[agent_name]
        if state.remaining_agents and state.remaining_agents[0] == agent_name:
            state.remaining_agents.pop(0)
        elif agent_name in state.remaining_agents:
            state.remaining_agents.remove(agent_name)
        state.current_step = agent.name
        self._log(f"{self._agent_label(agent.name)} 실행 시작")
        self.adapter.append_run_event(state.run_id, "agent.started", f"{agent.name} started.", node_name=agent.name)
        envelope = agent.run(state, self.runtime)
        self._record_agent_result(state, envelope)
        state.last_agent = envelope.agent_name
        state.completed_agents.append(envelope.agent_name)
        artifact_count = len(envelope.artifact_ids())
        self._log(f"{self._agent_label(agent.name)} 완료: status={envelope.status.value}, artifacts={artifact_count}")
        return {**graph_state, "state": state, "last_agent_result": envelope}

    def run_sql_agent(self, graph_state: dict[str, Any]) -> dict[str, Any]:
        return self._run_agent(graph_state, self.sql_agent.name)

    def run_eda_agent(self, graph_state: dict[str, Any]) -> dict[str, Any]:
        return self._run_agent(graph_state, self.eda_agent.name)

    def run_analysis_agent(self, graph_state: dict[str, Any]) -> dict[str, Any]:
        return self._run_agent(graph_state, self.analysis_agent.name)

    def run_visualization_agent(self, graph_state: dict[str, Any]) -> dict[str, Any]:
        return self._run_agent(graph_state, self.visualization_agent.name)

    def run_report_agent(self, graph_state: dict[str, Any]) -> dict[str, Any]:
        return self._run_agent(graph_state, self.report_agent.name)

    # Map a planned agent name to its explicit graph node.
    AGENT_NODES = {
        "sql_agent": "run_sql_agent",
        "eda_agent": "run_eda_agent",
        "analysis_agent": "run_analysis_agent",
        "visualization_agent": "run_visualization_agent",
        "report_agent": "run_report_agent",
    }

    def _agent_node_name(self, agent_name: str) -> str:
        return self.AGENT_NODES[agent_name]

    def central_validate(self, graph_state: dict[str, Any]) -> dict[str, Any]:
        state = graph_state["state"]
        upstream = graph_state["last_agent_result"]
        if upstream is None:
            raise RuntimeError("Validation requires an upstream agent result.")
        self._log(f"{self._agent_label(self.validation_agent.name)} 실행 시작: target={upstream.agent_name}")
        validation = self.validation_agent.run(state, self.runtime, upstream)
        state.validation_status = validation.status.value
        self._record_agent_result(state, validation)
        self._log(f"{self._agent_label(self.validation_agent.name)} 완료: status={validation.status.value}")
        return {**graph_state, "state": state, "last_validation_result": validation}

    def supervisor_gate(self, graph_state: dict[str, Any]) -> dict[str, Any]:
        state = graph_state["state"]
        upstream = graph_state["last_agent_result"]
        validation = graph_state["last_validation_result"]
        if upstream is None or validation is None:
            raise RuntimeError("Supervisor gate requires upstream and validation results.")

        if validation.status == AgentStatus.failed:
            if validation.retry_hint.retryable:
                state.terminal_state = SupervisorTerminalState.failed_with_recoverable_context
                self.adapter.update_run_status(
                    state.run_id,
                    RunStatus.failed,
                    metadata={"terminal_state": state.terminal_state.value, "retry_hint": validation.retry_hint.model_dump(mode="json")},
                )
                return {**graph_state, "state": state, "gate_result": "failed_with_recoverable_context"}
            state.terminal_state = SupervisorTerminalState.failed_terminal
            self.adapter.update_run_status(state.run_id, RunStatus.failed, metadata={"terminal_state": state.terminal_state.value})
            return {**graph_state, "state": state, "gate_result": "failed_terminal"}

        # Only DECIDE that approval is needed here. The approval request is
        # created downstream in ``approval_gate`` so this gate stays a pure
        # routing decision.
        if upstream.approval.required or validation.approval.required:
            return {**graph_state, "state": state, "gate_result": "needs_approval"}

        return {**graph_state, "state": state, "gate_result": "continue"}

    def approval_gate(self, graph_state: dict[str, Any]) -> dict[str, Any]:
        state = graph_state["state"]
        upstream = graph_state["last_agent_result"]
        validation = graph_state["last_validation_result"]
        if upstream is None or validation is None:
            raise RuntimeError("Approval gate requires upstream and validation results.")

        approval = self.adapter.request_approval(
            "mart.persist",
            "mart",
            {
                "reason": upstream.approval.reason or validation.approval.reason,
                "run_id": state.run_id,
                "source_artifact_ids": upstream.artifact_ids(),
                "plan": state.plan.model_dump(mode="json") if state.plan else {},
            },
        )
        state.approval_ids.append(approval.approval_id)
        state.terminal_state = SupervisorTerminalState.needs_user_approval
        state.current_step = "approval_gate"
        self.adapter.append_run_event(
            state.run_id,
            "approval.required",
            "Mart persistence requires approval.",
            node_name="approval_gate",
            approval_id=approval.approval_id,
        )
        self.adapter.update_run_status(
            state.run_id,
            RunStatus.waiting_approval,
            metadata={"terminal_state": state.terminal_state.value, "approval_id": approval.approval_id},
        )
        return {**graph_state, "state": state, "gate_result": "needs_user_approval"}

    def route_after_gate(self, graph_state: dict[str, Any]) -> str:
        state = graph_state["state"]
        gate_result = graph_state.get("gate_result")
        if gate_result == "continue":
            if state.remaining_agents:
                return self._agent_node_name(state.remaining_agents[0])
            return "finalize"
        if gate_result == "needs_approval":
            return "approval_gate"
        # failed_with_recoverable_context / failed_terminal
        return "end"

    def finalize(self, graph_state: dict[str, Any]) -> dict[str, Any]:
        state = graph_state["state"]
        state.current_step = "finalize"
        state.terminal_state = SupervisorTerminalState.completed
        self._log("Orchestration 완료")
        self.adapter.append_run_event(state.run_id, "supervisor.completed", "Supervisor completed.", node_name="finalize")
        self.adapter.update_run_status(state.run_id, RunStatus.succeeded, metadata={"terminal_state": state.terminal_state.value})
        return {**graph_state, "state": state}

    def resume_after_approval(self, state: OrchestrationState, approval_id: str) -> OrchestrationState:
        approval = self.adapter.get_approval_status(approval_id)
        if approval.status != ApprovalStatus.approved:
            state.terminal_state = SupervisorTerminalState.failed_with_recoverable_context
            state.error_state = {"approval_id": approval_id, "approval_status": approval.status.value}
            return state

        source_artifact_id = next(iter(state.artifact_ids.get("sql_agent", [])), "unknown")
        mart_ref = self.adapter.materialize_mart_metadata(
            state.run_id,
            mart_id=f"mart_{state.run_id}",
            source_sql_artifact_id=source_artifact_id,
            approval_id=approval_id,
            schema_json={"columns": []},
            refresh_policy="manual",
        )
        state.mart_id = f"mart_{state.run_id}"
        state.add_artifacts("mart_metadata", [mart_ref.artifact_id])
        return state

    def _agent_sequence(self) -> Iterable[object]:
        return tuple(self.agent_registry[name] for name in self.agent_registry)

    def _record_agent_result(self, state: OrchestrationState, envelope: AgentEnvelope) -> None:
        state.add_artifacts(envelope.agent_name, envelope.artifact_ids())
        self.adapter.append_run_event(
            state.run_id,
            "agent.completed",
            envelope.summary,
            node_name=envelope.agent_name,
            artifact_ids=envelope.artifact_ids(),
            metadata={
                "status": envelope.status.value,
                "next_handoff": envelope.next_handoff,
                "approval_required": envelope.approval.required,
            },
        )

    @staticmethod
    def _agent_label(agent_name: str) -> str:
        return AGENT_LOG_LABELS.get(agent_name, agent_name)

    @staticmethod
    def _log(message: str) -> None:
        print(f"✅ {message}", flush=True)

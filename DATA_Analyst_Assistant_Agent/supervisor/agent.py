from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from data_agent_backend.models.runs import RunStatus
from langgraph.types import Command

from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.shared.contracts import (
    OrchestrationState,
    SupervisorInterruptPayload,
    SupervisorRunResult,
    SupervisorTerminalState,
)
from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model
from DATA_Analyst_Assistant_Agent.supervisor.checkpoint import open_sqlite_checkpointer
from DATA_Analyst_Assistant_Agent.supervisor.graph import build_graph
from DATA_Analyst_Assistant_Agent.supervisor.candidate import commit_candidate
from DATA_Analyst_Assistant_Agent.supervisor.analysis_review import (
    AnalysisReviewResumePayload,
    validate_analysis_review_resume,
)
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    SupervisorState,
    empty_supervisor_state,
    normalize_supervisor_state,
    reject_pending_result,
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
    ) -> SupervisorRunResult:
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
            interrupt_payload = self._interrupt_payload_from_output(output)
            if interrupt_payload is not None:
                return self._update_run_from_interrupt(run.run_id, interrupt_payload)
            return self._update_run_from_terminal_output(run.run_id, output)
        except Exception as exc:
            self.adapter.update_run_status(run.run_id, RunStatus.failed, metadata={"error": str(exc)})
            raise

    def resume(self, thread_id: str, resume_payload: dict[str, Any]) -> Any:
        config = {"configurable": {"thread_id": thread_id}}
        with open_sqlite_checkpointer(self.checkpoint_path) as checkpointer:
            graph = self._build_runtime_graph(checkpointer)
            snapshot = graph.get_state(config) if hasattr(graph, "get_state") else None
            checkpoint_values = getattr(snapshot, "values", None)
            if snapshot is None:
                normalized_payload, resumed_from = self._validated_unchecked_resume(resume_payload)
                result = graph.invoke(Command(resume=normalized_payload), config)
            elif isinstance(checkpoint_values, dict):
                run_id = self._current_run_id_from_values(checkpoint_values)
                if not run_id:
                    raise ValueError(f"thread_id={thread_id!r}에 해당하는 checkpoint를 찾지 못했습니다.")
                interrupts = self._interrupt_payloads_from_snapshot(snapshot)
                if len(interrupts) > 1:
                    raise ValueError("활성 interrupt는 정확히 하나여야 합니다.")
                if interrupts:
                    interrupt_payload = interrupts[0]
                    if interrupt_payload.run_id != run_id or interrupt_payload.thread_id != thread_id:
                        raise ValueError("활성 interrupt와 checkpoint 식별자가 일치하지 않습니다.")
                    if interrupt_payload.type == "clarification":
                        pending_approval = checkpoint_values.get("pending_approval")
                        if (
                            interrupt_payload.node != "collect_clarification"
                            or checkpoint_values.get("next_action") == "collect_analysis_review"
                            or (
                                isinstance(pending_approval, dict)
                                and pending_approval.get("review_request") is not None
                            )
                        ):
                            raise ValueError("clarification interrupt와 checkpoint state가 불일치합니다.")
                        normalized_payload = {
                            "answer": self._validated_clarification_answer(resume_payload)
                        }
                        resumed_from = "clarification"
                    else:
                        normalized_payload = self._validated_analysis_review_resume(
                            resume_payload,
                            interrupt_payload,
                            checkpoint_values,
                        )
                        resumed_from = "analysis_review"
                    self.adapter.update_run_status(
                        run_id,
                        RunStatus.running,
                        metadata={"resumed_from": resumed_from},
                    )
                    result = graph.invoke(Command(resume=normalized_payload), config)
                else:
                    approval_payload = self._validated_approval_resume(resume_payload)
                    pending_approval = checkpoint_values.get("pending_approval")
                    if not isinstance(pending_approval, dict):
                        raise ValueError(f"thread_id={thread_id!r}는 승인 대기 상태가 아닙니다.")
                    if pending_approval.get("review_request") is not None:
                        raise ValueError("analysis review interrupt가 없는 구조화 승인은 재개할 수 없습니다.")
                    resumed_from = "approval"
                    self.adapter.update_run_status(
                        run_id,
                        RunStatus.running,
                        metadata={"resumed_from": resumed_from},
                    )
                    result = self._resume_synthetic_approval(
                        graph,
                        config,
                        approval_payload,
                        checkpoint_values,
                    )
                    if result is None:
                        raise ValueError(f"thread_id={thread_id!r}의 승인 대기 상태를 재개할 수 없습니다.")
            else:
                raise ValueError(f"thread_id={thread_id!r}에 해당하는 checkpoint를 찾지 못했습니다.")
        return self._update_run_status_from_resume_result(result)

    def get_checkpoint_state(self, thread_id: str) -> dict[str, Any] | None:
        """thread_id의 마지막 체크포인트를 읽기 전용으로 반환한다(그래프를 진행시키지 않음).

        분기(재분석) 트리거가 이전 단계의 artifact_ids/target_table을 알아내는 데 쓴다.
        """
        config = {"configurable": {"thread_id": thread_id}}
        with open_sqlite_checkpointer(self.checkpoint_path) as checkpointer:
            graph = self._build_runtime_graph(checkpointer)
            return self._checkpoint_values_from_graph(graph, config)

    @classmethod
    def _validated_unchecked_resume(
        cls,
        resume_payload: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        if set(resume_payload) == {"answer"}:
            return {"answer": cls._validated_clarification_answer(resume_payload)}, "clarification"
        if "approved" in resume_payload:
            return cls._validated_approval_resume(resume_payload), "approval"
        raise ValueError(
            "resume payload는 {'answer': '...'}, {'approved': True}, 또는 활성 analysis review 형식이어야 합니다."
        )

    @staticmethod
    def _validated_approval_resume(resume_payload: dict[str, Any]) -> dict[str, Any]:
        allowed = {"approved", "reason"}
        if not set(resume_payload).issubset(allowed) or "approved" not in resume_payload:
            raise ValueError("approval resume payload must include approved:boolean.")
        approved = resume_payload.get("approved")
        if not isinstance(approved, bool):
            raise ValueError("approval resume payload must include approved:boolean.")
        normalized: dict[str, Any] = {"approved": approved}
        reason = str(resume_payload.get("reason") or "").strip()
        if reason:
            normalized["reason"] = reason
        return normalized

    def _validated_analysis_review_resume(
        self,
        resume_payload: dict[str, Any],
        interrupt_payload: SupervisorInterruptPayload,
        checkpoint_values: dict[str, Any],
    ) -> dict[str, Any]:
        pending_approval = checkpoint_values.get("pending_approval")
        pending_result = checkpoint_values.get("pending_result")
        if interrupt_payload.node != "collect_analysis_review":
            raise ValueError("analysis review interrupt와 checkpoint state가 불일치합니다.")
        if not isinstance(pending_approval, dict) or not isinstance(pending_result, dict):
            raise ValueError("analysis review 대기 상태가 손상되었습니다.")
        if pending_approval.get("approval_id") != interrupt_payload.approval_id:
            raise ValueError("analysis review approval_id가 interrupt와 일치하지 않습니다.")
        if pending_approval.get("review_request") != interrupt_payload.review_request:
            raise ValueError("analysis review request가 interrupt와 checkpoint에서 다릅니다.")
        normalized = AnalysisReviewResumePayload.model_validate(resume_payload)
        decision = validate_analysis_review_resume(
            normalized,
            pending_approval=pending_approval,
            pending_candidate_id=str(pending_result.get("candidate_id") or ""),
            review_request=interrupt_payload.review_request or {},
        )
        expected_hashes = dict(pending_approval.get("content_hashes") or {})
        candidate_hashes = dict(pending_result.get("content_hashes") or {})
        if not expected_hashes or expected_hashes != candidate_hashes:
            raise ValueError("analysis review 대상의 content hash가 유효하지 않습니다.")
        for artifact_id, expected_hash in expected_hashes.items():
            try:
                artifact = self.adapter.get_artifact(artifact_id)
            except Exception as exc:
                raise ValueError("analysis review 대상의 content hash를 확인할 수 없습니다.") from exc
            if str(getattr(artifact, "content_hash", "") or "") != str(expected_hash):
                raise ValueError("analysis review 대상의 content hash가 변경되었습니다.")
        return AnalysisReviewResumePayload(
            approval_id=decision.approval_id,
            selected_option_id=decision.selection_response.selected_option_id,
            free_text=decision.selection_response.free_text,
        ).model_dump(mode="json", exclude_none=True)

    def _resume_clarification(self, thread_id: str, resume_payload: dict[str, Any]) -> Any:
        answer = self._validated_clarification_answer(resume_payload)
        config = {"configurable": {"thread_id": thread_id}}
        with open_sqlite_checkpointer(self.checkpoint_path) as checkpointer:
            graph = self._build_runtime_graph(checkpointer)
            checkpoint_values = self._checkpoint_values_from_graph(graph, config)
            if checkpoint_values is not None:
                run_id = self._current_run_id_from_values(checkpoint_values)
                if not run_id:
                    raise ValueError(f"thread_id={thread_id!r}에 해당하는 checkpoint를 찾지 못했습니다.")
                self.adapter.update_run_status(
                    run_id,
                    RunStatus.running,
                    metadata={"resumed_from": "clarification"},
                )
            result = graph.invoke(Command(resume={"answer": answer}), config)
        return self._update_run_status_from_resume_result(result)

    def _resume_synthetic_approval(
        self,
        graph: Any,
        config: dict[str, Any],
        resume_payload: dict[str, Any],
        checkpoint_values: dict[str, Any] | None = None,
    ) -> Any | None:
        if "approved" not in resume_payload:
            return None
        if not hasattr(graph, "get_state") or not hasattr(graph, "update_state"):
            return None

        values = checkpoint_values
        if values is None:
            values = self._checkpoint_values_from_graph(graph, config)
        if not isinstance(values, dict):
            return None

        pending_approval = values.get("pending_approval")
        if not isinstance(pending_approval, dict):
            return None

        if resume_payload.get("approved") is False:
            return self._reject_synthetic_approval(
                graph,
                config,
                values,
                str(resume_payload.get("reason") or "").strip(),
            )

        pending_result = values.get("pending_result")
        if isinstance(pending_result, dict) and pending_approval.get("candidate_id"):
            return self._resume_validated_candidate(
                graph,
                config,
                values,
                pending_result,
                pending_approval,
            )

        next_action = self._next_action_for_pending_agent(pending_approval.get("agent"))
        if next_action is None:
            return None

        updates = {
            "pending_approval": None,
            "terminal_state": "running",
            "next_action": next_action,
            "final_answer": "",
        }
        returned_config = graph.update_state(config, updates, as_node="decide_next_action")
        return graph.invoke(None, returned_config or config)

    def _reject_synthetic_approval(
        self,
        graph: Any,
        config: dict[str, Any],
        values: dict[str, Any],
        reason: str = "",
    ) -> Any:
        rejection_reason = reason or "사용자가 승인 요청을 거절했습니다."
        normalized = reject_pending_result(
            values,
            rejection_reason,
            {"approved": False, "reason": rejection_reason},
            event_type="approval.rejected",
        )
        updates = {
            **normalized,
            "pending_approval": None,
            "pending_validation": None,
            "terminal_state": "failed_with_recoverable_context",
            "next_action": "finalize",
            "final_answer": (
                "사용자가 승인 요청을 거절했습니다. 요청을 수정하거나 다른 기준으로 다시 실행할 수 있습니다."
            ),
        }
        returned_config = graph.update_state(config, updates, as_node="commit_candidate")
        return graph.invoke(None, returned_config or config)

    def _resume_validated_candidate(
        self,
        graph: Any,
        config: dict[str, Any],
        values: dict[str, Any],
        pending_result: dict[str, Any],
        pending_approval: dict[str, Any],
    ) -> Any:
        candidate_identity_matches = (
            pending_approval.get("candidate_id") == pending_result.get("candidate_id")
            and pending_approval.get("validation_id") == pending_result.get("validation_id")
        )
        expected_hashes = dict(pending_approval.get("content_hashes") or {})
        hashes_match = candidate_identity_matches and bool(expected_hashes)
        if hashes_match:
            for artifact_id, expected_hash in expected_hashes.items():
                try:
                    record = self.adapter.get_artifact(artifact_id)
                except Exception:
                    hashes_match = False
                    break
                if str(getattr(record, "content_hash", "") or "") != str(expected_hash):
                    hashes_match = False
                    break

        normalized = normalize_supervisor_state(values)
        if hashes_match:
            updates = commit_candidate(
                values,
                self.adapter,
                approval_granted=True,
            )
            updates.update(
                {
                    "pending_approval": None,
                    "terminal_state": "running",
                    "final_answer": "",
                }
            )
            returned_config = graph.update_state(config, updates, as_node="commit_candidate")
            return graph.invoke(None, returned_config or config)

        events = list(normalized.get("run_events", []))
        events.append(
            {
                "type": "approval.invalidated",
                "candidate_id": pending_result.get("candidate_id"),
                "validation_id": pending_result.get("validation_id"),
                "reason": "승인 대상의 식별자 또는 content hash가 변경되었습니다.",
            }
        )
        append_event = getattr(self.adapter, "append_run_event", None)
        if append_event is not None:
            append_event(
                str(values.get("current_run_id") or ""),
                "approval.invalidated",
                "승인 대상의 식별자 또는 content hash가 변경되었습니다.",
                node_name="supervisor",
                metadata=events[-1],
            )
        updates = {
            **normalized,
            "pending_approval": None,
            "pending_validation": None,
            "terminal_state": "running",
            "final_answer": "",
            "next_action": self._next_action_for_pending_agent(
                (pending_result.get("result") or {}).get("agent")
            ) or "finalize",
            "run_events": events,
        }
        anchor = "execute_subagent"
        updates["current_step"] = anchor
        returned_config = graph.update_state(config, updates, as_node=anchor)
        return graph.invoke(None, returned_config or config)

    @staticmethod
    def _next_action_for_pending_agent(agent_name: Any) -> str | None:
        return {
            "sql_agent": "call_sql_agent",
            "eda_agent": "call_eda_agent",
            "analysis_agent": "call_analysis_agent",
        }.get(str(agent_name))

    def _update_run_status_from_resume_result(self, result: Any) -> Any:
        interrupt_payload = self._interrupt_payload_from_output(result)
        if interrupt_payload is not None:
            return self._update_run_from_interrupt(interrupt_payload.run_id, interrupt_payload)

        if not isinstance(result, dict):
            return result

        run_id = result.get("current_run_id")
        if not isinstance(run_id, str) or not run_id:
            return result

        terminal_state = result.get("terminal_state")
        if not terminal_state or terminal_state == "running":
            return result

        try:
            return self._update_run_from_terminal_output(run_id, result)
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
        if datasource_id is not None:
            return datasource_id
        getter = getattr(self.adapter, "get_default_datasource_id", None)
        if getter is None:
            return None
        return getter()

    def _resolve_catalog_summary(self, datasource_id: str | None) -> dict[str, Any] | None:
        if datasource_id is None:
            return None
        getter = getattr(self.adapter, "get_catalog_summary", None)
        if getter is None:
            return None
        return getter(datasource_id)

    def _update_run_from_terminal_output(self, run_id: str, output: SupervisorState) -> SupervisorRunResult:
        orchestration_state = to_orchestration_state(output)
        self.adapter.update_run_status(
            run_id,
            self._run_status_for_terminal(orchestration_state.terminal_state),
            metadata={"terminal_state": output.get("terminal_state")},
        )
        return SupervisorRunResult(kind="state", state=orchestration_state)

    def _update_run_from_interrupt(
        self,
        run_id: str,
        interrupt_payload: SupervisorInterruptPayload,
    ) -> SupervisorRunResult:
        self.adapter.update_run_status(
            run_id,
            RunStatus.waiting_input,
            metadata={
                "interrupt_type": interrupt_payload.type,
                "node": interrupt_payload.node,
            },
        )
        append_event = getattr(self.adapter, "append_run_event", None)
        if append_event is not None:
            append_event(
                run_id,
                "human_input.required",
                interrupt_payload.question,
                node_name=interrupt_payload.node,
                metadata=interrupt_payload.model_dump(mode="json"),
            )
        return SupervisorRunResult(kind="interrupt", interrupt=interrupt_payload)

    @staticmethod
    def _interrupt_payload_from_output(output: Any) -> SupervisorInterruptPayload | None:
        if not isinstance(output, dict) or "__interrupt__" not in output:
            return None
        interrupts = output.get("__interrupt__") or ()
        if not interrupts:
            return None
        if len(interrupts) != 1:
            raise ValueError("활성 interrupt는 정확히 하나여야 합니다.")
        first_interrupt = interrupts[0]
        raw_payload = getattr(first_interrupt, "value", first_interrupt)
        return SupervisorInterruptPayload.model_validate(raw_payload)

    @staticmethod
    def _interrupt_payloads_from_snapshot(snapshot: Any) -> list[SupervisorInterruptPayload]:
        payloads: list[SupervisorInterruptPayload] = []
        for task in getattr(snapshot, "tasks", ()) or ():
            for item in getattr(task, "interrupts", ()) or ():
                raw_payload = getattr(item, "value", item)
                payloads.append(SupervisorInterruptPayload.model_validate(raw_payload))
        return payloads

    @staticmethod
    def _validated_clarification_answer(resume_payload: dict[str, Any]) -> str:
        if set(resume_payload) != {"answer"}:
            raise ValueError("clarification resume payload는 정확히 {'answer': '...'}여야 합니다.")
        answer = str(resume_payload.get("answer") or "").strip()
        if not answer:
            raise ValueError("clarification resume payload의 answer는 비어 있을 수 없습니다.")
        return answer

    @staticmethod
    def _checkpoint_values_from_graph(graph: Any, config: dict[str, Any]) -> dict[str, Any] | None:
        if not hasattr(graph, "get_state"):
            return None
        snapshot = graph.get_state(config)
        values = getattr(snapshot, "values", None)
        if isinstance(values, dict):
            return values
        return {}

    @staticmethod
    def _current_run_id_from_values(values: dict[str, Any]) -> str | None:
        run_id = values.get("current_run_id")
        return run_id if isinstance(run_id, str) and run_id else None

    @staticmethod
    def _run_status_for_terminal(terminal_state: SupervisorTerminalState | None) -> RunStatus:
        if terminal_state == SupervisorTerminalState.completed:
            return RunStatus.succeeded
        if terminal_state == SupervisorTerminalState.needs_user_approval:
            return RunStatus.waiting_approval
        # 백엔드에는 clarification 전용 상태가 없으므로 나머지 terminal 상태는 실패로 기록한다.
        return RunStatus.failed


SQLAgentSupervisor = SupervisorAgent

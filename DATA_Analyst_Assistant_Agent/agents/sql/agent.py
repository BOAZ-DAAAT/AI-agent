from __future__ import annotations

import csv
import datetime
import decimal
import io
import json
from typing import Any

from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.shared.contracts import (
    AgentEnvelope,
    AgentStatus,
    LocalCheck,
    OrchestrationState,
    OlistTemplateId,
    OlistTemplateKind,
    ValidationBlock,
    ValidationFinding,
)
from DATA_Analyst_Assistant_Agent.agents.sql.validation_artifact import build_validation_summary_payload


def _looks_temporal_column(name: str) -> bool:
    normalized = name.casefold()
    return any(token in normalized for token in ("date", "time", "month", "year", "timestamp"))


class SQLAgent:
    name = "sql_agent"

    def run(self, state: OrchestrationState, runtime: AgentRuntime) -> AgentEnvelope:
        if self._is_contract_repair_mode(state):
            return self._repair_analysis_data_contract_only(state, runtime)
        result = self._run_main_sql_agent(state)
        return self._envelope_from_main_result(state, runtime, result)

    @staticmethod
    def _is_contract_repair_mode(state: OrchestrationState) -> bool:
        retry_context = state.plan.retry_context if state.plan else state.retry_context
        if not isinstance(retry_context, dict):
            return False
        last_failure = retry_context.get("last_failure") or {}
        return (
            retry_context.get("mode") == "contract_only"
            and retry_context.get("suggested_action") == "repair_analysis_data_contract"
            and isinstance(last_failure, dict)
            and last_failure.get("reason_code") == "analysis_contract_invalid"
        )

    def _repair_analysis_data_contract_only(
        self,
        state: OrchestrationState,
        runtime: AgentRuntime,
    ) -> AgentEnvelope:
        context = runtime.context(state, node_name=self.name, tool_name="sql_agent.contract_repair")
        if state.plan is None:
            return AgentEnvelope(
                status=AgentStatus.failed,
                agent_name=self.name,
                summary="분석 입력 계약을 보강할 SQL 계획이 없습니다.",
                validation=ValidationBlock(local_checks=[
                    LocalCheck(
                        name="analysis_data_contract_repaired",
                        passed=False,
                        severity="error",
                        detail="analysis_plan is missing.",
                    )
                ]),
                retry_hint={
                    "retryable": False,
                    "suggested_action": "stop_and_surface_error",
                    "reason_code": "missing_analysis_plan",
                    "details": {},
                },
                error="analysis_plan is missing.",
            )

        generated_sql = state.plan.generated_sql or state.plan.source_sql or state.generated_sql
        sql_draft = {
            "sql": generated_sql,
            "target_table": state.plan.target_table,
            "source_tables": state.plan.source_tables,
            "business_grain": state.plan.business_grain,
        }
        contract = self._repair_contract_from_existing_plan(state.plan, sql_draft, generated_sql)
        state.plan.analysis_data_contract = contract
        state.generated_sql = generated_sql
        state.planner_mode = state.plan.planner_mode
        state.error_state = {}

        ref = runtime.adapter.register_artifact(
            state.run_id,
            "file",
            content_text=json.dumps(
                {
                    "analysis_data_contract": contract,
                    "generated_sql_preserved": generated_sql,
                    "target_table": state.plan.target_table,
                    "mode": "contract_only",
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            filename=f"analysis_data_contract_repair_{state.run_id}.json",
            created_by_tool="sql_agent.contract_repair",
            context=context,
            metadata={"kind": "analysis_data_contract", "mode": "contract_only"},
            preview={
                "row_grain": contract.get("row_grain"),
                "target_table": contract.get("target_table"),
                "derived_column_count": len(contract.get("derived_columns") or []),
            },
        )
        return AgentEnvelope(
            status=AgentStatus.success,
            agent_name=self.name,
            summary="기존 SQL/데이터마트를 재생성하지 않고 분석 입력 계약만 보강했습니다.",
            artifact_refs=[ref],
            validation=ValidationBlock(local_checks=[
                LocalCheck(
                    name="analysis_data_contract_repaired",
                    passed=bool(str(contract.get("row_grain") or "").strip()),
                    severity="error" if not str(contract.get("row_grain") or "").strip() else "info",
                    detail="analysis_data_contract row_grain repaired.",
                )
            ]),
            retry_hint={
                "retryable": False,
                "suggested_action": "call_analysis_agent",
                "reason_code": "analysis_data_contract_repaired",
                "details": {"mode": "contract_only"},
            },
        )

    def _run_main_sql_agent(self, state: OrchestrationState) -> dict[str, Any]:
        from DATA_Analyst_Assistant_Agent.agents.sql.graph import (
            SQL_MAX_REPAIR_RETRIES,
            SQL_MAX_RETRIES,
            build_app,
        )

        app = build_app()
        catalog_summary = state.catalog_summary or {}
        retry_context = state.plan.retry_context if state.plan else None
        required_derivations = (
            [
                item.model_dump(mode="json")
                for item in state.plan.required_derivations
            ]
            if state.plan is not None
            else []
        )
        analysis_heuristics = (
            [
                item.model_dump(mode="json")
                for item in state.plan.analysis_heuristics
            ]
            if state.plan is not None
            else []
        )
        if state.plan is not None and required_derivations:
            state.plan.route_kind = "comprehensive"
            state.plan.requires_mart_review = True
            state.plan.sql_generation_source = "semantic_llm"
            state.plan.sql_template_id = None
            state.plan.sql_template_kind = None
        supervisor_plan_context = {}
        if state.plan is not None:
            supervisor_plan_context = {
                "goal": state.plan.goal,
                "route_kind": state.plan.route_kind,
                "metric": state.plan.metric,
                "dimension": state.plan.dimension,
                "filters": state.plan.filters,
                "requires_mart_review": state.plan.requires_mart_review,
                "sql_generation_source": state.plan.sql_generation_source,
                "sql_template_id": (
                    state.plan.sql_template_id.value if state.plan.sql_template_id else None
                ),
                "sql_template_kind": (
                    state.plan.sql_template_kind.value if state.plan.sql_template_kind else None
                ),
                "sql_template_parameters": state.plan.sql_template_parameters.model_dump(),
            }
            if state.plan.query_rules:
                supervisor_plan_context["query_rules"] = state.plan.query_rules
        clarification_request = ""
        if retry_context:
            feedback = (retry_context.get("agent_feedback") or {}).get("sql_agent")
            if isinstance(feedback, dict) and (feedback.get("reason") or feedback.get("missing_evidence")):
                reason = feedback.get("reason") or ""
                missing = feedback.get("missing_evidence") or []
                missing_text = (
                    f" 누락된 근거: {', '.join(str(item) for item in missing)}." if missing else ""
                )
                clarification_request = f"이전 시도가 검증에 실패했습니다: {reason}.{missing_text}".strip()
            else:
                retry_message = retry_context.get("message") or ""
                retry_query = retry_context.get("query") or ""
                retry_step = retry_context.get("step") or state.current_step
                clarification_request = (
                    f"이전 {retry_step} 실패 원인: {retry_message}. "
                    f"문제가 된 SQL: {retry_query}"
                ).strip()
        planner_selection_reason = state.goal or ""
        if supervisor_plan_context:
            planner_selection_reason = (
                f"{planner_selection_reason}\n\nSupervisor analysis_plan:\n"
                f"{json.dumps(supervisor_plan_context, ensure_ascii=False, indent=2)}"
            ).strip()

        return app.invoke(
            {
                "user_question": state.user_query,
                "required_db_schema": json.dumps(catalog_summary, ensure_ascii=False) if catalog_summary else "",
                "clarification_request": clarification_request,
                "planner_selection_reason": planner_selection_reason,
                "required_derivations": required_derivations,
                "analysis_heuristics": analysis_heuristics,
                "schema_text": json.dumps(catalog_summary, ensure_ascii=False) if catalog_summary else "",
                "integrity_text": "",
                "integrity_dataset_name": state.datasource_id or "default",
                "integrity_preplan": {},
                "integrity_refresh": {},
                "schema_refresh": {},
                "question_plan": {},
                "final_table_plan": {},
                "planning_stages": {},
                "plan": {},
                "mart_design": {},
                "sql_draft": {},
                "previous_sql_draft": {},
                "sql_result": None,
                "statement_results": [],
                "row_count": 0,
                "precheck_result": None,
                "postcheck_result": None,
                "mart_quality_result": {},
                "validation": {},
                "validation_findings": [],
                "retry_hint": {},
                "validation_summary": {},
                "retry_count": 0,
                "repair_retry_count": 0,
                "feedback": "",
                "error": "",
                "generation_source": (
                    "olist_template"
                    if state.plan is not None and state.plan.sql_template_id is not None
                    else "semantic_llm"
                ),
                "sql_generation_source": (
                    "olist_template"
                    if state.plan is not None and state.plan.sql_template_id is not None
                    else "semantic_llm"
                ),
                "sql_template_id": (
                    state.plan.sql_template_id.value
                    if state.plan is not None and state.plan.sql_template_id is not None
                    else None
                ),
                "sql_template_kind": (
                    state.plan.sql_template_kind.value
                    if state.plan is not None and state.plan.sql_template_kind is not None
                    else None
                ),
                "sql_template_parameters": (
                    state.plan.sql_template_parameters.model_dump()
                    if state.plan is not None
                    else {}
                ),
                "max_retries": SQL_MAX_RETRIES,
                "max_repair_retries": SQL_MAX_REPAIR_RETRIES,
                "generation_failure_reason": "",
                "generation_context_diagnostics": [],
                "failed_statement_index": None,
                "failed_statement_sql": "",
                "failed_sql_component": None,
                "execution_error_info": {},
                "classification": "none",
                "repair_strategy": "none",
                "repair_attempted": False,
                "repair_validation_result": {
                    "result": "not_attempted",
                    "reason_code": "not_attempted",
                    "details": {},
                },
                "final_answer": "",
            }
        )

    def _envelope_from_main_result(
        self,
        state: OrchestrationState,
        runtime: AgentRuntime,
        result: dict[str, Any],
    ) -> AgentEnvelope:
        context = runtime.context(state, node_name=self.name, tool_name="sql_agent.lang_graph")
        sql_draft = result.get("sql_draft") or {}
        generated_sql = sql_draft.get("sql") or ""
        generation_source = self._normalized_generation_source(
            result.get("sql_generation_source") or result.get("generation_source")
        )
        template_id_value = result.get("sql_template_id") or (
            state.plan.sql_template_id.value
            if state.plan is not None and state.plan.sql_template_id is not None
            else None
        )
        template_id = OlistTemplateId(template_id_value) if template_id_value else None
        template_kind_value = result.get("sql_template_kind") or (
            state.plan.sql_template_kind.value
            if state.plan is not None and state.plan.sql_template_kind is not None
            else None
        )
        template_kind = OlistTemplateKind(template_kind_value) if template_kind_value else None
        template_parameters = (
            state.plan.sql_template_parameters.model_dump(mode="json")
            if state.plan is not None
            else {}
        )
        state.generated_sql = generated_sql
        state.planner_mode = "deterministic" if template_id is not None else "llm"
        # comprehensive(마트) 경로면 마트 테이블 참조를 plan에 실어 하류로 넘긴다.
        # 하류(EDA/분석)는 이 이름으로 DB에서 마트를 직접 조회한다. simple 경로면 target_table 없음.
        target_table = sql_draft.get("target_table") if sql_draft.get("sql_type") != "select" else None
        # 하류(EDA/분석)가 GE 정합성 스코핑·grain 교차검증에 쓸 원천 테이블/선언 grain을 함께 승격.
        source_tables = [str(t) for t in (sql_draft.get("source_tables") or []) if t]
        business_grain = sql_draft.get("business_grain") or None
        if state.plan is not None:
            state.plan.generated_sql = generated_sql
            state.plan.source_sql = generated_sql
            state.plan.planner_mode = state.planner_mode
            state.plan.sql_generation_source = generation_source
            state.plan.sql_template_id = template_id
            state.plan.sql_template_kind = template_kind
            state.plan.target_table = target_table or None
            state.plan.source_tables = source_tables
            state.plan.business_grain = business_grain
            state.plan.mart_design = result.get("mart_design") or {}
            state.plan.analysis_data_contract = self._analysis_data_contract(
                result.get("mart_design") or {},
                sql_draft,
                generated_sql,
                required_derivations=[
                    item.model_dump(mode="json")
                    for item in state.plan.required_derivations
                ],
                analysis_heuristics=[
                    item.model_dump(mode="json")
                    for item in state.plan.analysis_heuristics
                ],
            )

        plan_payload = {
            "plan": result.get("plan") or {},
            "planning_stages": result.get("planning_stages") or {
                "question_plan": result.get("question_plan") or {},
                "final_table_plan": result.get("final_table_plan") or {},
            },
            "mart_design": result.get("mart_design") or {},
            "sql_draft": sql_draft,
            "statement_results": result.get("statement_results") or [],
            "validation": result.get("validation") or {},
            "generation_source": generation_source,
            "sql_generation_source": generation_source,
            "sql_template_id": template_id.value if template_id else None,
            "sql_template_kind": template_kind.value if template_kind else None,
            "sql_template_parameters": template_parameters,
            "generation_failure_reason": result.get("generation_failure_reason") or "",
            "failed_statement_index": result.get("failed_statement_index"),
            "failed_statement_sql": result.get("failed_statement_sql") or "",
            "failed_sql_component": result.get("failed_sql_component"),
            "execution_error_info": result.get("execution_error_info") or {},
            "classification": result.get("classification") or "none",
            "repair_strategy": result.get("repair_strategy") or "none",
            "repair_attempted": bool(result.get("repair_attempted")),
            "repair_validation_result": result.get("repair_validation_result") or {},
            "final_answer": result.get("final_answer") or "",
            "source": "main/sql_agent/lang graph",
        }
        plan_ref = runtime.adapter.register_artifact(
            state.run_id,
            "file",
            content_text=json.dumps(plan_payload, ensure_ascii=False, indent=2, default=str),
            filename=f"sql_lang_graph_result_{state.run_id}.json",
            created_by_tool="sql_agent.lang_graph",
            context=context,
            metadata={
                "kind": "sql_lang_graph_result",
                "source": "main/sql_agent/lang graph",
                "sql_type": sql_draft.get("sql_type"),
                "generation_source": generation_source,
                "sql_generation_source": generation_source,
                "sql_template_id": template_id.value if template_id else None,
                "sql_template_kind": template_kind.value if template_kind else None,
                "target_table": target_table,
            },
            preview={
                "question_type": (result.get("plan") or {}).get("question_type"),
                "task_type": (result.get("plan") or {}).get("task_type"),
                "validation": (result.get("validation") or {}).get("result"),
                "final_answer": result.get("final_answer") or "",
                "generation_source": generation_source,
                "sql_generation_source": generation_source,
                "sql_template_id": template_id.value if template_id else None,
                "sql_template_kind": template_kind.value if template_kind else None,
            },
        )
        sql_plan_ref = runtime.adapter.register_artifact(
            state.run_id,
            "file",
            content_text=json.dumps(plan_payload, ensure_ascii=False, indent=2, default=str),
            filename=f"sql_plan_{state.run_id}.json",
            created_by_tool="sql_agent.lang_graph",
            context=context,
            parent_ids=[plan_ref.artifact_id],
            metadata={
                "kind": "sql_plan",
                "source": "main/sql_agent/lang graph",
                "generation_source": generation_source,
                "sql_generation_source": generation_source,
                "sql_template_id": template_id.value if template_id else None,
                "sql_template_kind": template_kind.value if template_kind else None,
                "target_table": target_table,
            },
            preview={
                "route_kind": (result.get("plan") or {}).get("route_kind"),
                "generation_source": generation_source,
                "sql_generation_source": generation_source,
                "sql_template_id": template_id.value if template_id else None,
                "sql_template_kind": template_kind.value if template_kind else None,
            },
        )
        sql_ref = runtime.adapter.register_artifact(
            state.run_id,
            "sql_query",
            content_text=generated_sql,
            filename=f"generated_sql_{state.run_id}.sql",
            created_by_tool="sql_agent.lang_graph",
            context=context,
            parent_ids=[plan_ref.artifact_id],
            metadata={
                "kind": "generated_sql",
                "source": "main/sql_agent/lang graph",
                "generation_source": generation_source,
                "sql_generation_source": generation_source,
                "sql_template_id": template_id.value if template_id else None,
                "sql_template_kind": template_kind.value if template_kind else None,
                "target_table": target_table,
            },
            preview={
                "sql": generated_sql,
                "sql_type": sql_draft.get("sql_type"),
                "generation_source": generation_source,
                "sql_generation_source": generation_source,
                "sql_template_id": template_id.value if template_id else None,
                "sql_template_kind": template_kind.value if template_kind else None,
            },
        )

        result_csv, result_columns, result_row_count = self._main_sql_result_to_csv(result.get("sql_result"))
        result_ref = runtime.adapter.register_artifact(
            state.run_id,
            "sql_result",
            content_text=result_csv,
            filename=f"sql_lang_graph_result_{state.run_id}.csv",
            created_by_tool="sql_agent.lang_graph",
            context=context,
            parent_ids=[sql_ref.artifact_id],
            lineage_edge_type="query_result_of",
            metadata={
                "kind": "sql_result",
                "source": "main/sql_agent/lang graph",
                "generation_source": generation_source,
                "sql_generation_source": generation_source,
                "sql_template_id": template_id.value if template_id else None,
                "sql_template_kind": template_kind.value if template_kind else None,
                "target_table": target_table,
            },
            preview={
                "row_count": result_row_count,
                "columns": result_columns,
                "sample_rows": self._sample_rows_for_preview(result.get("sql_result")),
                "generation_source": generation_source,
                "sql_generation_source": generation_source,
                "sql_template_id": template_id.value if template_id else None,
                "sql_template_kind": template_kind.value if template_kind else None,
            },
        )
        validation_payload = build_validation_summary_payload(result)
        validation_ref = runtime.adapter.register_artifact(
            state.run_id,
            "file",
            content_text=json.dumps(validation_payload, ensure_ascii=False, indent=2, default=str),
            filename=f"sql_validation_summary_{state.run_id}.json",
            created_by_tool="sql_agent.lang_graph",
            context=context,
            parent_ids=[plan_ref.artifact_id, sql_ref.artifact_id],
            metadata={
                "kind": "ge_table_validation_json",
                "source": "main/sql_agent/lang graph",
                "generation_source": generation_source,
                "sql_generation_source": generation_source,
                "sql_template_id": template_id.value if template_id else None,
                "sql_template_kind": template_kind.value if template_kind else None,
            },
            preview={
                "validation_result": (result.get("validation") or {}).get("result"),
                "reason_code": ((result.get("retry_hint") or {}).get("reason_code")),
                "generation_source": generation_source,
                "sql_generation_source": generation_source,
                "sql_template_id": template_id.value if template_id else None,
                "sql_template_kind": template_kind.value if template_kind else None,
            },
        )

        validation = result.get("validation") or {}
        retry_hint = self._final_retry_hint(state, result, generation_source, template_id)
        if validation.get("result") == "invalid":
            state.error_state = {
                "code": retry_hint.get("reason_code") or "SQL_GENERATION_OR_EXECUTION_FAILED",
                "message": validation.get("reason", "SQL LangGraph validation failed."),
                "query": generated_sql,
                "datasource_id": state.datasource_id,
                "step": self.name,
                "hint": validation.get("feedback", ""),
            }
            if state.plan is not None:
                state.plan.retry_context = {
                    "code": state.error_state["code"],
                    "message": state.error_state["message"],
                    "query": state.error_state["query"],
                    "datasource_id": state.datasource_id,
                    "step": self.name,
                    "hint": state.error_state.get("hint", ""),
                    "details": retry_hint.get("details", {}),
                }
        else:
            state.error_state = {}

        checks = [
            LocalCheck(
                name="main_sql_agent_generated_sql",
                passed=bool(generated_sql.strip()),
                severity="error" if not generated_sql.strip() else "info",
                detail="SQL LangGraph generated SQL." if generated_sql.strip() else "SQL LangGraph did not return SQL.",
            ),
            LocalCheck(
                name="main_sql_agent_result_rows",
                passed=result_row_count >= 0,
                severity="info",
                detail=f"row_count={result_row_count}.",
            ),
        ]
        fallback_used = False
        if generation_source == "failed":
            checks.append(
                LocalCheck(
                    name="main_sql_agent_generation_source",
                    passed=False,
                    severity="error",
                    detail=f"generation_source=failed reason={result.get('generation_failure_reason') or 'none'}",
                )
            )
        elif result.get("repair_attempted"):
            checks.append(
                LocalCheck(
                    name="main_sql_agent_generation_source",
                    passed=True,
                    severity="info",
                    detail="generation_source=semantic_llm repair_attempted=true",
                )
            )
        if validation.get("result") == "invalid":
            checks.append(
                LocalCheck(
                    name="main_sql_agent_validation",
                    passed=False,
                    severity="error",
                    detail=validation.get("reason", "SQL LangGraph validation failed."),
                )
            )

        has_error = any(not check.passed and check.severity == "error" for check in checks)
        findings = [self._validation_finding(item) for item in result.get("validation_findings") or []]
        return AgentEnvelope(
            status=AgentStatus.failed if has_error else AgentStatus.success,
            agent_name=self.name,
            summary=result.get("final_answer") or "SQL LangGraph agent completed.",
            artifact_refs=[result_ref, plan_ref, sql_plan_ref, sql_ref, validation_ref],
            validation=ValidationBlock(local_checks=checks, findings=findings),
            retry_hint={
                "retryable": bool(retry_hint.get("retryable", has_error)),
                "suggested_action": retry_hint.get("suggested_action", "fix_sql"),
                "reason_code": retry_hint.get("reason_code", "main_sql_agent_validation" if has_error else "none"),
                "details": retry_hint.get("details", {}),
            },
            fallback_used=fallback_used,
        )

    @staticmethod
    def _normalized_generation_source(value: Any) -> str:
        if value in {"llm", "repair", "semantic_llm"}:
            return "semantic_llm"
        if value in {"olist_template", "failed"}:
            return str(value)
        return "semantic_llm"

    @classmethod
    def _final_retry_hint(
        cls,
        state: OrchestrationState,
        result: dict[str, Any],
        generation_source: str,
        template_id: OlistTemplateId | None,
    ) -> dict[str, Any]:
        retry_hint = dict(result.get("retry_hint") or {})
        validation = result.get("validation") or {}
        if validation.get("result") != "invalid":
            return retry_hint
        if template_id is not None:
            return {
                **retry_hint,
                "retryable": False,
                "suggested_action": "stop_and_surface_error",
                "reason_code": retry_hint.get("reason_code") or "olist_template_failure",
            }

        reason_code = str(retry_hint.get("reason_code") or "")
        technical_codes = {"execution_error", "missing_table", "missing_column", "olist_template_failure"}
        if reason_code in technical_codes:
            return {
                **retry_hint,
                "retryable": False,
                "suggested_action": "stop_and_surface_error",
            }
        if generation_source in {"semantic_llm", "failed"}:
            details = dict(retry_hint.get("details") or {})
            details.update(
                {
                    "clarification_question": cls._clarification_question(state),
                    "failure_reason": validation.get("reason") or result.get("generation_failure_reason") or "SQL 생성 및 검증 실패",
                }
            )
            return {
                **retry_hint,
                "retryable": False,
                "suggested_action": "clarify",
                "reason_code": reason_code or "semantic_sql_failed",
                "details": details,
            }
        return retry_hint

    @staticmethod
    def _clarification_question(state: OrchestrationState) -> str:
        metric = state.plan.metric if state.plan is not None else None
        dimension = state.plan.dimension if state.plan is not None else None
        examples = []
        if not metric:
            examples.append("핵심 지표")
        if not dimension:
            examples.append("집계 단위")
        examples.append("분석 기간")
        return f"분석 범위를 다시 정할 수 있도록 {', '.join(examples)}를 구체적으로 알려주세요."

    @staticmethod
    def _validation_finding(item: dict[str, Any]) -> ValidationFinding:
        severity = str(item.get("severity") or "info")
        retryable = bool(item.get("retryable", False))
        explicit_disposition = str(item.get("disposition") or "").strip()
        if explicit_disposition:
            disposition = explicit_disposition
        elif severity == "error":
            disposition = "error"
        elif severity == "warning":
            disposition = "warning"
        else:
            disposition = "diagnostic"
        return ValidationFinding(
            code=str(item.get("code") or item.get("category") or "sql_validation"),
            source=str(item.get("source") or "sql_langgraph"),
            severity=severity,
            disposition=disposition,
            message=str(item.get("message") or item.get("detail") or "SQL validation finding"),
            retryable=retryable,
            suggested_action=str(item.get("suggested_action") or ""),
            details=dict(item.get("details") or {}),
        )

    @staticmethod
    def _analysis_data_contract(
        mart_design: dict[str, Any],
        sql_draft: dict[str, Any],
        generated_sql: str,
        *,
        required_derivations: list[dict[str, Any]] | None = None,
        analysis_heuristics: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        column_plan = list(mart_design.get("column_plan") or [])
        grain_columns = list(mart_design.get("grain_columns") or [])
        derived_columns = [
            {
                "output_column": item.get("output_column"),
                "role": item.get("role"),
                "source_columns": list(item.get("source_columns") or []),
                "calculation_type": item.get("calculation_type"),
                "calculation_rule": item.get("calculation_rule"),
                "aggregation_method": item.get("aggregation_method"),
            }
            for item in column_plan
            if isinstance(item, dict)
        ]
        aggregation_rules = [
            item
            for item in derived_columns
            if str(item.get("aggregation_method") or "none").lower() != "none"
        ]
        required_derivations = list(required_derivations or [])
        analysis_heuristics = list(analysis_heuristics or [])
        output_columns = {
            str(item.get("output_column") or "").strip()
            for item in column_plan
            if isinstance(item, dict)
        }
        implemented_derivations = [
            {
                **item,
                "output_column": str(
                    item.get("preferred_name") or item.get("name") or ""
                ).strip(),
            }
            for item in required_derivations
            if str(item.get("preferred_name") or item.get("name") or "").strip()
            in output_columns
        ]
        unimplemented_names = {
            str(item.get("name") or "").strip()
            for item in mart_design.get("unimplemented_derivations") or []
            if isinstance(item, dict)
        }
        unimplemented_derivations: list[dict[str, Any]] = []
        for item in required_derivations:
            preferred_name = str(
                item.get("preferred_name") or item.get("name") or ""
            ).strip()
            if preferred_name not in unimplemented_names:
                continue
            implementation = next(
                (
                    candidate
                    for candidate in mart_design.get("unimplemented_derivations") or []
                    if isinstance(candidate, dict)
                    and str(candidate.get("name") or "").strip() == preferred_name
                ),
                {},
            )
            unimplemented_derivations.append(
                {
                    **item,
                    "output_column": preferred_name,
                    "reason": implementation.get("reason"),
                    "required_columns": list(
                        implementation.get("required_columns") or []
                    ),
                }
            )
        return {
            "target_table": sql_draft.get("target_table") or mart_design.get("mart_name"),
            "row_grain": mart_design.get("grain") or sql_draft.get("business_grain") or "",
            "grain_columns": grain_columns,
            "entity_keys": [
                column for column in grain_columns if not _looks_temporal_column(column)
            ],
            "time_basis": [
                item
                for item in derived_columns
                if _looks_temporal_column(str(item.get("output_column") or ""))
                or any(_looks_temporal_column(str(source)) for source in item.get("source_columns") or [])
            ],
            "derived_columns": derived_columns,
            "aggregation_rules": aggregation_rules,
            "deduplication_keys": list(mart_design.get("deduplication_keys") or []),
            "source_tables": list(mart_design.get("source_tables") or sql_draft.get("source_tables") or []),
            "source_grains": dict(mart_design.get("source_grains") or {}),
            "safe_interpretations": [
                "선언된 row_grain 기준에서만 데이터마트를 해석합니다.",
                "집계 수준의 연관성을 개별 주문 수준 인과로 해석하지 않습니다.",
                "계약과 표본이 충분히 뒷받침하지 않으면 추세 검정은 기술적 해석으로 제한합니다.",
            ],
            "required_derivations": required_derivations,
            "analysis_heuristics": analysis_heuristics,
            "implemented_derivations": implemented_derivations,
            "unimplemented_derivations": unimplemented_derivations,
            "generated_sql": generated_sql,
        }

    @classmethod
    def _repair_contract_from_existing_plan(
        cls,
        plan: Any,
        sql_draft: dict[str, Any],
        generated_sql: str,
    ) -> dict[str, Any]:
        existing = dict(getattr(plan, "analysis_data_contract", None) or {})
        mart_design = dict(getattr(plan, "mart_design", None) or {})
        contract = cls._analysis_data_contract(
            mart_design,
            sql_draft,
            generated_sql,
            required_derivations=[
                item.model_dump(mode="json")
                for item in getattr(plan, "required_derivations", [])
            ],
            analysis_heuristics=[
                item.model_dump(mode="json")
                for item in getattr(plan, "analysis_heuristics", [])
            ],
        )
        inferred_columns = cls._infer_column_lineage_from_sql(generated_sql)
        if inferred_columns and not contract.get("derived_columns"):
            contract["derived_columns"] = inferred_columns
        elif inferred_columns:
            known = {
                str(item.get("output_column") or "")
                for item in contract.get("derived_columns") or []
                if isinstance(item, dict)
            }
            contract["derived_columns"] = [
                *(contract.get("derived_columns") or []),
                *[item for item in inferred_columns if item["output_column"] not in known],
            ]

        group_columns = cls._infer_group_columns_from_sql(generated_sql)
        for key, value in existing.items():
            if value not in (None, "", [], {}):
                contract[key] = value
        if not str(contract.get("row_grain") or "").strip():
            contract["row_grain"] = getattr(plan, "business_grain", None) or ", ".join(group_columns)
        if not contract.get("grain_columns"):
            contract["grain_columns"] = group_columns
        if not contract.get("entity_keys"):
            contract["entity_keys"] = [
                column for column in contract.get("grain_columns", []) if not _looks_temporal_column(column)
            ]
        if not contract.get("deduplication_keys"):
            contract["deduplication_keys"] = list(contract.get("grain_columns") or [])
        if not contract.get("time_basis"):
            contract["time_basis"] = [
                item
                for item in contract.get("derived_columns", [])
                if isinstance(item, dict)
                and (
                    _looks_temporal_column(str(item.get("output_column") or ""))
                    or any(_looks_temporal_column(str(source)) for source in item.get("source_columns") or [])
                )
            ]
        contract["target_table"] = contract.get("target_table") or getattr(plan, "target_table", None)
        contract["source_tables"] = contract.get("source_tables") or list(getattr(plan, "source_tables", []) or [])
        contract["generated_sql"] = generated_sql
        return contract

    @staticmethod
    def _infer_column_lineage_from_sql(sql: str) -> list[dict[str, Any]]:
        if not str(sql or "").strip():
            return []
        try:
            import sqlglot
            from sqlglot import exp

            tree = sqlglot.parse_one(sql, error_level="ignore")
            select = tree.find(exp.Select) if tree is not None else None
            if select is None:
                return []
            items: list[dict[str, Any]] = []
            for expression in select.expressions:
                alias = expression.alias_or_name
                if not alias:
                    continue
                source_columns = sorted({
                    column.sql(dialect="mysql")
                    for column in expression.find_all(exp.Column)
                })
                agg = next(expression.find_all(exp.AggFunc), None)
                items.append(
                    {
                        "output_column": alias,
                        "source_columns": source_columns or [alias],
                        "calculation_type": (
                            "aggregate"
                            if agg is not None
                            else "derived" if expression.find(exp.Func) or expression.find(exp.Case) else "passthrough"
                        ),
                        "calculation_rule": expression.sql(dialect="mysql"),
                        "aggregation_method": agg.key.upper() if agg is not None else "none",
                    }
                )
            return items
        except Exception:
            return []

    @staticmethod
    def _infer_group_columns_from_sql(sql: str) -> list[str]:
        try:
            import sqlglot
            from sqlglot import exp

            tree = sqlglot.parse_one(sql, error_level="ignore")
            group = tree.find(exp.Group) if tree is not None else None
            if group is None:
                return []
            return [item.sql(dialect="mysql").split(".")[-1] for item in group.expressions]
        except Exception:
            return []

    @staticmethod
    def _main_sql_result_to_csv(rows: Any) -> tuple[str, list[str], int]:
        if not rows:
            return "", [], 0

        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            if hasattr(row, "_mapping"):
                normalized_rows.append(SQLAgent._json_safe(dict(row._mapping)))
            elif isinstance(row, dict):
                normalized_rows.append(SQLAgent._json_safe(row))
            elif isinstance(row, (list, tuple)):
                normalized_rows.append(SQLAgent._json_safe({f"col_{idx + 1}": value for idx, value in enumerate(row)}))
            else:
                normalized_rows.append({"value": SQLAgent._json_safe(row)})

        columns = list(normalized_rows[0].keys())
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized_rows)
        return output.getvalue(), columns, len(normalized_rows)

    @staticmethod
    def _sample_rows_for_preview(rows: Any) -> list[dict[str, Any]]:
        if not rows:
            return []
        sample_rows = []
        for row in list(rows)[:5]:
            if hasattr(row, "_mapping"):
                sample_rows.append(SQLAgent._json_safe(dict(row._mapping)))
            elif isinstance(row, dict):
                sample_rows.append(SQLAgent._json_safe(row))
            elif isinstance(row, (list, tuple)):
                sample_rows.append(SQLAgent._json_safe({f"col_{idx + 1}": value for idx, value in enumerate(row)}))
            else:
                sample_rows.append({"value": SQLAgent._json_safe(row)})
        return sample_rows

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, decimal.Decimal):
            return float(value)
        if isinstance(value, (datetime.date, datetime.datetime)):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(key): SQLAgent._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [SQLAgent._json_safe(item) for item in value]
        return value

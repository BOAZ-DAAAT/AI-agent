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
    ValidationBlock,
    ValidationFinding,
)
from DATA_Analyst_Assistant_Agent.agents.sql.validation_artifact import build_validation_summary_payload


class SQLAgent:
    name = "sql_agent"

    def run(self, state: OrchestrationState, runtime: AgentRuntime) -> AgentEnvelope:
        result = self._run_main_sql_agent(state)
        return self._envelope_from_main_result(state, runtime, result)

    def _run_main_sql_agent(self, state: OrchestrationState) -> dict[str, Any]:
        from DATA_Analyst_Assistant_Agent.agents.sql.graph import build_app

        app = build_app()
        catalog_summary = state.catalog_summary or {}
        retry_context = state.plan.retry_context if state.plan else None
        clarification_request = ""
        if retry_context:
            retry_message = retry_context.get("message") or ""
            retry_query = retry_context.get("query") or ""
            retry_step = retry_context.get("step") or state.current_step
            clarification_request = (
                f"이전 {retry_step} 실패 원인: {retry_message}. "
                f"문제가 된 SQL: {retry_query}"
            ).strip()

        return app.invoke(
            {
                "user_question": state.user_query,
                "required_db_schema": json.dumps(catalog_summary, ensure_ascii=False) if catalog_summary else "",
                "clarification_request": clarification_request,
                "planner_selection_reason": state.goal or "",
                "schema_text": json.dumps(catalog_summary, ensure_ascii=False) if catalog_summary else "",
                "integrity_text": "",
                "integrity_dataset_name": state.datasource_id or "default",
                "integrity_preplan": {},
                "integrity_refresh": {},
                "plan": {},
                "mart_design": {},
                "sql_draft": {},
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
                "max_retries": 2,
                "feedback": "",
                "error": "",
                "generation_source": "llm",
                "fallback_reason": "",
                "failed_statement_index": None,
                "failed_statement_sql": "",
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
        state.generated_sql = generated_sql
        state.planner_mode = "llm"
        # comprehensive(마트) 경로면 마트 테이블 참조를 plan에 실어 하류로 넘긴다.
        # 하류(EDA/분석)는 이 이름으로 DB에서 마트를 직접 조회한다. simple 경로면 target_table 없음.
        target_table = sql_draft.get("target_table") if sql_draft.get("sql_type") != "select" else None
        # 하류(EDA/분석)가 GE 정합성 스코핑·grain 교차검증에 쓸 원천 테이블/선언 grain을 함께 승격.
        source_tables = [str(t) for t in (sql_draft.get("source_tables") or []) if t]
        business_grain = sql_draft.get("business_grain") or None
        if state.plan is not None:
            state.plan.generated_sql = generated_sql
            state.plan.source_sql = generated_sql
            state.plan.planner_mode = "llm"
            state.plan.target_table = target_table or None
            state.plan.source_tables = source_tables
            state.plan.business_grain = business_grain

        plan_payload = {
            "plan": result.get("plan") or {},
            "mart_design": result.get("mart_design") or {},
            "sql_draft": sql_draft,
            "statement_results": result.get("statement_results") or [],
            "validation": result.get("validation") or {},
            "generation_source": result.get("generation_source") or "llm",
            "fallback_reason": result.get("fallback_reason") or "",
            "failed_statement_index": result.get("failed_statement_index"),
            "failed_statement_sql": result.get("failed_statement_sql") or "",
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
                "generation_source": result.get("generation_source") or "llm",
            },
            preview={
                "question_type": (result.get("plan") or {}).get("question_type"),
                "task_type": (result.get("plan") or {}).get("task_type"),
                "validation": (result.get("validation") or {}).get("result"),
                "final_answer": result.get("final_answer") or "",
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
            metadata={"kind": "sql_plan", "source": "main/sql_agent/lang graph"},
            preview={"route_kind": (result.get("plan") or {}).get("route_kind")},
        )
        sql_ref = runtime.adapter.register_artifact(
            state.run_id,
            "sql_query",
            content_text=generated_sql,
            filename=f"generated_sql_{state.run_id}.sql",
            created_by_tool="sql_agent.lang_graph",
            context=context,
            parent_ids=[plan_ref.artifact_id],
            metadata={"kind": "generated_sql", "source": "main/sql_agent/lang graph"},
            preview={"sql": generated_sql, "sql_type": sql_draft.get("sql_type")},
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
            metadata={"kind": "sql_result", "source": "main/sql_agent/lang graph"},
            preview={
                "row_count": result_row_count,
                "columns": result_columns,
                "sample_rows": self._sample_rows_for_preview(result.get("sql_result")),
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
            metadata={"kind": "ge_table_validation_json", "source": "main/sql_agent/lang graph"},
            preview={
                "validation_result": (result.get("validation") or {}).get("result"),
                "reason_code": ((result.get("retry_hint") or {}).get("reason_code")),
            },
        )

        validation = result.get("validation") or {}
        retry_hint = result.get("retry_hint") or {}
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
        fallback_used = bool(result.get("generation_source") in {"fallback", "hard_fallback"} and result.get("retry_count", 0) > 0)
        if result.get("generation_source") in {"fallback", "hard_fallback"}:
            checks.append(
                LocalCheck(
                    name="main_sql_agent_generation_source",
                    passed=not fallback_used,
                    severity="warning" if fallback_used else "info",
                    detail=f"generation_source={result.get('generation_source')} reason={result.get('fallback_reason') or 'none'}",
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
    def _validation_finding(item: dict[str, Any]) -> ValidationFinding:
        severity = str(item.get("severity") or "info")
        retryable = bool(item.get("retryable", False))
        if retryable:
            disposition = "retry_required"
        elif severity == "error":
            disposition = "blocking"
        elif severity == "warning":
            disposition = "limitation"
        else:
            disposition = "advisory"
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

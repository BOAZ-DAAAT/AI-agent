from __future__ import annotations

import csv
import datetime
import decimal
import io
import json
import sys
from pathlib import Path
from typing import Any

from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.shared.contracts import AgentEnvelope, AgentStatus, LocalCheck, OrchestrationState, ValidationBlock


class SQLAgent:
    name = "sql_agent"

    def run(self, state: OrchestrationState, runtime: AgentRuntime) -> AgentEnvelope:
        result = self._run_main_sql_agent(state)
        return self._envelope_from_main_result(state, runtime, result)

    def _run_main_sql_agent(self, state: OrchestrationState) -> dict[str, Any]:
        from DATA_Analyst_Assistant_Agent.agents.sql.graph import build_app

        app = build_app()
        return app.invoke(
            {
                "user_question": state.user_query,
                "schema_text": "",
                "integrity_text": "",
                "plan": {},
                "mart_design": {},
                "sql_draft": {},
                "sql_result": None,
                "row_count": 0,
                "precheck_result": None,
                "postcheck_result": None,
                "mart_quality_result": {},
                "validation": {},
                "retry_count": 0,
                "max_retries": 2,
                "feedback": "",
                "error": "",
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
        if state.plan is not None:
            state.plan.generated_sql = generated_sql
            state.plan.source_sql = generated_sql
            state.plan.planner_mode = "llm"

        plan_payload = {
            "plan": result.get("plan") or {},
            "mart_design": result.get("mart_design") or {},
            "sql_draft": sql_draft,
            "validation": result.get("validation") or {},
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
            },
            preview={
                "question_type": (result.get("plan") or {}).get("question_type"),
                "task_type": (result.get("plan") or {}).get("task_type"),
                "validation": (result.get("validation") or {}).get("result"),
                "final_answer": result.get("final_answer") or "",
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

        validation = result.get("validation") or {}
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
        return AgentEnvelope(
            status=AgentStatus.failed if has_error else AgentStatus.success,
            agent_name=self.name,
            summary=result.get("final_answer") or "SQL LangGraph agent completed.",
            artifact_refs=[result_ref, plan_ref, sql_ref],
            validation=ValidationBlock(local_checks=checks),
            retry_hint={
                "retryable": has_error,
                "suggested_action": "fix_sql",
                "reason_code": "main_sql_agent_validation" if has_error else "none",
            },
            next_handoff="validation_agent",
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

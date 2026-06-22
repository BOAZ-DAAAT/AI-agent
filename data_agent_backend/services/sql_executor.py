from __future__ import annotations

from time import perf_counter

try:
    import sqlglot
    from sqlglot import expressions as exp
except ModuleNotFoundError:  # pragma: no cover - optional local execution dependency
    sqlglot = None
    exp = None

from data_agent_backend.config import BackendConfig
from data_agent_backend.models.artifacts import ArtifactRegisterRequest, ArtifactRef, ArtifactType
from data_agent_backend.models.common import BackendError, JsonDict
from data_agent_backend.models.contexts import PolicyContext
from data_agent_backend.models.datasource import DatasourceCredential, DatasourceType
from data_agent_backend.services.artifact_registry import ArtifactRegistry
from data_agent_backend.services.datasource_service import DatasourceService
from data_agent_backend.services.policy_engine import PolicyEngine


BLOCKED_SQL_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "MERGE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "PRAGMA",
    "ATTACH",
    "DETACH",
    "INSTALL",
    "LOAD",
    "COPY",
    "EXPORT",
    "CALL",
}


class SQLExecutor:
    def __init__(
        self,
        config: BackendConfig,
        registry: ArtifactRegistry,
        policy_engine: PolicyEngine,
        datasource_service: DatasourceService,
    ) -> None:
        self.config = config
        self.registry = registry
        self.policy_engine = policy_engine
        self.datasource_service = datasource_service

    def run_sql_query(
        self,
        query: str,
        run_id: str,
        datasource_id: str | None = None,
        credential: DatasourceCredential | None = None,
        context: PolicyContext | None = None,
        row_limit: int | None = None,
    ) -> ArtifactRef:
        context = context or PolicyContext(run_id=run_id)
        resolved_datasource_id = self.datasource_service.resolve_datasource_id(datasource_id)
        record = self.datasource_service.get(resolved_datasource_id)
        row_limit = self.config.default_sql_row_limit if row_limit is None else row_limit
        validation = self.validate_sql(query, row_limit, record.type)
        decision = self.policy_engine.evaluate(
            "sql.run",
            resolved_datasource_id,
            {
                "datasource_id": resolved_datasource_id,
                "row_limit": row_limit,
                "max_row_limit": self.config.max_sql_row_limit_without_approval,
                **validation,
            },
            context,
        )
        if not decision.allowed:
            code = "APPROVAL_REQUIRED" if decision.requires_approval else "POLICY_BLOCKED"
            raise BackendError(code, decision.reason, {"decision_id": decision.decision_id})
        query_artifact = self.registry.register_artifact(
            ArtifactRegisterRequest(
                run_id=run_id,
                type=ArtifactType.sql_query,
                content_text=query,
                filename="query.sql",
                created_by_tool=context.tool_name or "sql_run_query",
                thread_id=context.thread_id,
                project_id=context.project_id,
                metadata={"datasource_id": resolved_datasource_id, "row_limit": row_limit},
            ),
            context,
        )
        rows, columns, csv_text = self.datasource_service.query_datasource(resolved_datasource_id, credential, query, row_limit)
        result_artifact = self.registry.register_artifact(
            ArtifactRegisterRequest(
                run_id=run_id,
                type=ArtifactType.sql_result,
                content_text=csv_text,
                filename="result.csv",
                created_by_tool=context.tool_name or "sql_run_query",
                thread_id=context.thread_id,
                project_id=context.project_id,
                parent_ids=[query_artifact.artifact_id],
                lineage_edge_type="query_result_of",
                metadata={"datasource_id": resolved_datasource_id, "row_limit": row_limit, "returned_rows": len(rows)},
                preview={"row_count": len(rows), "columns": columns, "sample_rows": [dict(zip(columns, row)) for row in rows[:5]]},
            ),
            context,
        )
        return result_artifact.ref()

    def run_analysis_query(
        self,
        query: str,
        run_id: str,
        datasource_id: str | None = None,
        credential: DatasourceCredential | None = None,
        context: PolicyContext | None = None,
        row_limit: int | None = None,
    ) -> JsonDict:
        context = context or PolicyContext(run_id=run_id, tool_name="db_run_analysis_query")
        context = context.model_copy(update={"run_id": run_id, "tool_name": "db_run_analysis_query"})
        resolved_datasource_id = self.datasource_service.resolve_datasource_id(datasource_id)
        record = self.datasource_service.get(resolved_datasource_id)
        row_limit = self.config.default_sql_row_limit if row_limit is None else row_limit
        validation = self.validate_sql(query, row_limit, record.type)
        decision = self.policy_engine.evaluate(
            "sql.run",
            resolved_datasource_id,
            {
                "datasource_id": resolved_datasource_id,
                "row_limit": row_limit,
                "max_row_limit": self.config.max_sql_row_limit_without_approval,
                **validation,
            },
            context,
        )
        if decision.requires_approval:
            raise BackendError("APPROVAL_REQUIRED", decision.reason, {"decision_id": decision.decision_id})
        if not decision.allowed:
            raise BackendError("POLICY_BLOCKED", decision.reason, {"decision_id": decision.decision_id})

        query_artifact = self.registry.register_artifact(
            ArtifactRegisterRequest(
                run_id=run_id,
                type=ArtifactType.sql_query,
                content_text=query,
                filename="query.sql",
                created_by_tool="db_run_analysis_query",
                thread_id=context.thread_id,
                project_id=context.project_id,
                metadata={"datasource_id": resolved_datasource_id, "row_limit": row_limit},
            ),
            context,
        )
        started = perf_counter()
        rows, columns, csv_text = self.datasource_service.query_datasource(resolved_datasource_id, credential, query, row_limit)
        runtime_ms = int((perf_counter() - started) * 1000)
        preview_rows = [dict(zip(columns, row)) for row in rows[:50]]
        preview = {
            "columns": columns,
            "rows": preview_rows,
            "truncated": len(rows) > len(preview_rows),
        }
        result_artifact = self.registry.register_artifact(
            ArtifactRegisterRequest(
                run_id=run_id,
                type=ArtifactType.sql_result,
                content_text=csv_text,
                filename="result.csv",
                created_by_tool="db_run_analysis_query",
                thread_id=context.thread_id,
                project_id=context.project_id,
                parent_ids=[query_artifact.artifact_id],
                lineage_edge_type="query_result_of",
                metadata={"datasource_id": resolved_datasource_id, "row_limit": row_limit, "returned_rows": len(rows)},
                preview={
                    "row_count": len(rows),
                    "columns": columns,
                    "sample_rows": preview_rows,
                },
            ),
            context,
        )
        artifact_ref = result_artifact.ref().model_dump(mode="json")
        artifact_ref["format"] = "csv"
        return {
            "artifact_ref": artifact_ref,
            "preview": preview,
            "profile": {
                "returned_rows": len(rows),
                "preview_rows": len(preview_rows),
                "column_count": len(columns),
            },
            "execution": {
                "datasource_id": resolved_datasource_id,
                "tool_name": "db_run_analysis_query",
                "row_limit": row_limit,
                "runtime_ms": runtime_ms,
            },
            "warnings": [],
        }

    def validate_sql(self, query: str, row_limit: int, datasource_type: DatasourceType | str = DatasourceType.mysql) -> JsonDict:
        stripped = query.strip()
        if not stripped:
            return {"blocked": True, "reason": "SQL query is empty."}
        if self._has_multiple_statements(stripped):
            return {"blocked": True, "reason": "Multiple-statement SQL is blocked."}
        upper_tokens = {token.upper() for token in stripped.replace(";", " ").replace("\n", " ").split()}
        blocked = sorted(upper_tokens & BLOCKED_SQL_KEYWORDS)
        if blocked:
            return {"blocked": True, "reason": f"Blocked SQL keyword(s): {', '.join(blocked)}."}
        if sqlglot is not None and exp is not None:
            try:
                parsed = sqlglot.parse_one(stripped, read=self._sqlglot_dialect(datasource_type))
            except Exception as exc:
                return {"blocked": True, "reason": f"SQL parse failed: {exc}"}
            if not isinstance(parsed, (exp.Select, exp.Union, exp.With)):
                return {"blocked": True, "reason": "Only read-only SELECT queries are allowed."}
        elif stripped.lstrip().split()[0].upper() not in {"SELECT", "WITH"}:
            return {"blocked": True, "reason": "Only read-only SELECT queries are allowed."}
        if row_limit <= 0:
            return {"blocked": True, "reason": "row_limit must be positive."}
        return {"blocked": False}

    def _has_multiple_statements(self, query: str) -> bool:
        trimmed = query.strip()
        if ";" not in trimmed:
            return False
        return trimmed.rstrip().rstrip(";").find(";") != -1

    def _sqlglot_dialect(self, datasource_type: DatasourceType | str) -> str:
        value = datasource_type.value if hasattr(datasource_type, "value") else str(datasource_type)
        if value == DatasourceType.postgresql.value:
            return "postgres"
        if value == DatasourceType.sqlite.value:
            return "sqlite"
        return "mysql"


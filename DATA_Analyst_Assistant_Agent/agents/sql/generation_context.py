"""SQL 생성 LLM에 전달할 최소 내부 컨텍스트를 구성한다."""

from __future__ import annotations

import json
import re
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from DATA_Analyst_Assistant_Agent.agents.sql._runtime import ALLOWED_MART_SCHEMA
from DATA_Analyst_Assistant_Agent.agents.sql.planner_support import schema_tables
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState, MartDesign


class _ContextModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    user_question: str = Field(min_length=1)
    filters: list[str]
    selected_tables: list[str] = Field(min_length=1)
    required_columns: list[str] = Field(min_length=1)
    business_keys: dict[str, str]
    previous_feedback: str
    schema_context: dict[str, Any] | str = Field(alias="schema", serialization_alias="schema")
    integrity_failures: list[str] | str

    @property
    def schema(self) -> dict[str, Any] | str:
        """직렬화 계약 이름과 동일한 schema 값을 제공한다."""
        return self.schema_context


class SimpleSQLGenerationContext(_ContextModel):
    dimensions: list[str]
    required_aggregations: list[str]
    expected_result_shape: Any = None


class GenerationFinalGrain(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    grain_columns: list[str] = Field(min_length=1)
    deduplication_keys: list[str] = Field(min_length=1)


class GenerationColumnPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_column: str = Field(min_length=1)
    source_columns: list[str] = Field(min_length=1)
    role: Literal["dimension", "measure", "attribute"]
    calculation_type: Literal["passthrough", "derived"]
    calculation_rule: str = Field(min_length=1)
    aggregation_method: Literal["none", "SUM", "COUNT", "COUNT_DISTINCT", "MIN", "MAX", "AVG", "DEDUPLICATE"]


class GenerationMetricSupport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_name: str = Field(min_length=1)
    calculation_grain: list[str]
    required_mart_columns: list[str] = Field(min_length=1)
    downstream_calculation: str = Field(min_length=1)


class ComprehensiveSQLGenerationContext(_ContextModel):
    target_table: str = Field(min_length=1)
    source_grains: dict[str, list[str]] = Field(min_length=1)
    final_grain: GenerationFinalGrain
    column_plan: list[GenerationColumnPlan] = Field(min_length=1)
    metric_support: list[GenerationMetricSupport] = Field(min_length=1)
    aggregation_policy: Literal["preserve_common_grain", "aggregate_to_common_grain"]


SQLGenerationContext: TypeAlias = SimpleSQLGenerationContext | ComprehensiveSQLGenerationContext


class GenerationContextResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    context: SQLGenerationContext
    context_json: str
    diagnostics: dict[str, Any]


_QUALIFIED_IDENTIFIER = re.compile(r"(?<![\w`])`?([A-Za-z_][\w$]*)`?\s*\.\s*`?([A-Za-z_][\w$]*)`?")
_INTEGRITY_LINE = re.compile(
    r"^- \[(?P<status>[A-Za-z_]+)\]\s+(?P<location>[^:]+):\s*(?P<detail>.*)$"
)
_FAIL_STATUSES = {"FAIL", "FAILED", "ERROR", "ACTION_REQUIRED"}
_INTEGRITY_HEADERS = {
    "Integrity context (only FAILED checks):",
    "Integrity context (only failures/warnings/stale checks):",
}


def _normalize_name(value: Any) -> str:
    return str(value or "").strip().strip("`").split(".")[-1]


def _compact_description(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    first_line = value.splitlines()[0] if value.splitlines() else ""
    return " ".join(first_line.split())[:160]


def _column_index(tables: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for table_name, table in tables.items():
        if not isinstance(table, dict):
            continue
        columns = table.get("columns")
        if not isinstance(columns, list):
            columns = []
        index[table_name] = {
            str(column.get("name")): column
            for column in columns
            if isinstance(column, dict) and str(column.get("name") or "").strip()
        }
    return index


def _reference_values(
    plan: dict[str, Any],
    design: MartDesign | None,
) -> list[tuple[str, str | None]]:
    references: list[tuple[str, str | None]] = []
    references.extend((str(value), None) for value in plan.get("required_columns") or [])
    references.extend(
        (str(value), _normalize_name(table))
        for table, value in (plan.get("business_keys") or {}).items()
    )
    if design is not None:
        for table, columns in design.source_grains.items():
            references.extend((str(value), _normalize_name(table)) for value in columns)
        for column in design.column_plan:
            references.extend((str(value), None) for value in column.source_columns)
    return references


def _resolve_reference(
    value: str,
    table_context: str | None,
    column_index: dict[str, dict[str, dict[str, Any]]],
) -> tuple[str, str] | None:
    qualified = _QUALIFIED_IDENTIFIER.fullmatch(value.strip())
    if qualified:
        table_name, column_name = qualified.groups()
        if table_name in column_index and column_name in column_index[table_name]:
            return table_name, column_name
        return None

    column_name = value.strip().strip("`")
    if not re.fullmatch(r"[A-Za-z_][\w$]*", column_name):
        return None
    if table_context:
        if column_name in column_index.get(table_context, {}):
            return table_context, column_name
        return None
    matches = [table for table, columns in column_index.items() if column_name in columns]
    return (matches[0], column_name) if len(matches) == 1 else None


def _filter_references(filters: list[str], column_index: dict[str, dict[str, dict[str, Any]]]) -> list[tuple[str, str | None]]:
    references: list[tuple[str, str | None]] = []
    all_columns = set().union(*(set(columns) for columns in column_index.values())) if column_index else set()
    for filter_text in filters:
        occupied: list[tuple[int, int]] = []
        for match in _QUALIFIED_IDENTIFIER.finditer(filter_text):
            references.append((match.group(0), None))
            occupied.append(match.span())
        unqualified_text = "".join(
            " " if any(start <= index < end for start, end in occupied) else character
            for index, character in enumerate(filter_text)
        )
        for token in re.findall(r"(?<![\w`])`?([A-Za-z_][\w$]*)`?(?![\w`])", unqualified_text):
            if token in all_columns:
                references.append((token, None))
    return references


def _foreign_key_endpoints(table_name: str, foreign_key: Any) -> tuple[tuple[str, str], tuple[str, str]] | None:
    if not isinstance(foreign_key, dict):
        return None
    local = foreign_key.get("column") or foreign_key.get("local_column")
    local_columns = foreign_key.get("columns") or foreign_key.get("local_columns")
    if not local and isinstance(local_columns, list) and len(local_columns) == 1:
        local = local_columns[0]
    references = foreign_key.get("references")
    if isinstance(references, dict):
        remote_table = references.get("table") or references.get("table_name")
        remote = references.get("column") or references.get("column_name")
    else:
        remote_table = (
            foreign_key.get("referenced_table")
            or foreign_key.get("reference_table")
            or foreign_key.get("foreign_table")
            or foreign_key.get("referred_table")
        )
        remote = (
            foreign_key.get("referenced_column")
            or foreign_key.get("reference_column")
            or foreign_key.get("foreign_column")
            or foreign_key.get("referred_column")
        )
    if not local or not remote_table or not remote:
        return None
    return (table_name, _normalize_name(local)), (_normalize_name(remote_table), _normalize_name(remote))


def _project_schema(
    schema_text: str,
    scoped_tables: list[str],
    selected_tables: list[str],
    references: list[tuple[str, str | None]],
    filters: list[str],
) -> tuple[dict[str, Any] | str, set[tuple[str, str]], bool, str | None]:
    try:
        parsed = json.loads(schema_text)
    except Exception:
        return schema_text, set(), True, "malformed_schema_json"
    if not isinstance(parsed, dict):
        return schema_text, set(), True, "malformed_schema_json"

    all_tables = schema_tables(parsed)
    if not isinstance(all_tables, dict) or any(table not in all_tables for table in scoped_tables):
        return schema_text, set(), True, "missing_table_reference"
    tables = {table: all_tables[table] for table in scoped_tables}
    index = _column_index(tables)
    references = [*references, *_filter_references(filters, index)]
    resolved: set[tuple[str, str]] = set()
    for value, table_context in references:
        result = _resolve_reference(value, table_context, index)
        if result is None:
            return schema_text, set(), True, "ambiguous_or_missing_column_reference"
        resolved.add(result)

    for table_name in selected_tables:
        table = tables.get(table_name)
        if not isinstance(table, dict):
            return schema_text, set(), True, "missing_table_reference"
        primary_key = table.get("primary_key") or []
        if isinstance(primary_key, str):
            primary_key = [primary_key]
        for column_name in primary_key:
            normalized = _normalize_name(column_name)
            if normalized not in index.get(table_name, {}):
                return schema_text, set(), True, "missing_primary_key_reference"
            resolved.add((table_name, normalized))

    relevant_fks: dict[str, list[Any]] = {table: [] for table in scoped_tables}
    scoped_set = set(scoped_tables)
    for table_name, table in tables.items():
        foreign_keys = table.get("foreign_keys") if isinstance(table, dict) else []
        if not isinstance(foreign_keys, list):
            continue
        for foreign_key in foreign_keys:
            endpoints = _foreign_key_endpoints(table_name, foreign_key)
            if endpoints is None:
                continue
            local, remote = endpoints
            if remote[0] not in scoped_set:
                continue
            if local[1] not in index.get(local[0], {}) or remote[1] not in index.get(remote[0], {}):
                return schema_text, set(), True, "missing_foreign_key_reference"
            resolved.update((local, remote))
            relevant_fks[table_name].append({
                "column": local[1],
                "referenced_table": remote[0],
                "referenced_column": remote[1],
            })

    if any(not any(table_name == ref_table for ref_table, _ in resolved) for table_name in scoped_tables):
        return schema_text, set(), True, "empty_table_projection"

    projected_tables: dict[str, Any] = {}
    for table_name, table in tables.items():
        projected_columns = []
        for column_name, column in index[table_name].items():
            if (table_name, column_name) not in resolved:
                continue
            projected_columns.append({
                "name": column_name,
                "type": column.get("type"),
                "nullable": column.get("nullable"),
                "description": _compact_description(column.get("description")),
            })
        primary_key = table.get("primary_key") or []
        if isinstance(primary_key, str):
            primary_key = [primary_key]
        projected_tables[table_name] = {
            "primary_key": [
                _normalize_name(column) for column in primary_key
                if (table_name, _normalize_name(column)) in resolved
            ],
            "foreign_keys": relevant_fks[table_name],
            "columns": projected_columns,
        }
    wrapper = {"tables": projected_tables} if isinstance(parsed.get("tables"), dict) else projected_tables
    return wrapper, resolved, False, None


def _project_integrity(
    integrity_text: str,
    selected_tables: list[str],
    referenced_columns: set[tuple[str, str]],
) -> tuple[list[str] | str, bool, str | None]:
    if not integrity_text.strip():
        return [], False, None
    selected = set(selected_tables)
    retained: list[str] = []
    for raw_line in integrity_text.splitlines():
        line = raw_line.strip()
        if not line or line in _INTEGRITY_HEADERS:
            continue
        match = _INTEGRITY_LINE.fullmatch(line)
        if match is None:
            return integrity_text, True, "malformed_integrity_line"
        status = match.group("status").upper()
        if status not in _FAIL_STATUSES:
            continue
        location = match.group("location").strip().strip("`")
        if "." in location:
            table_name, column_name = location.rsplit(".", 1)
            table_name, column_name = _normalize_name(table_name), _normalize_name(column_name)
            if column_name.lower() == "table-level":
                if table_name in selected:
                    retained.append(line)
            elif (table_name, column_name) in referenced_columns:
                retained.append(line)
        elif _normalize_name(location) in selected:
            retained.append(line)
    return retained, False, None


def _component_lengths(values: dict[str, Any]) -> dict[str, int]:
    lengths: dict[str, int] = {}
    for key, value in values.items():
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        lengths[key] = len(text)
    return lengths


def build_generation_context(
    state: AgentState,
    route_kind: str,
    previous_feedback: str,
) -> GenerationContextResult:
    plan = state.get("plan") or {}
    selected_tables = list(dict.fromkeys(_normalize_name(table) for table in plan.get("selected_join_tables") or []))
    if not selected_tables:
        raise ValueError("selected_join_tables가 비어 있습니다")

    design: MartDesign | None = None
    scoped_tables = list(selected_tables)
    if route_kind == "comprehensive":
        design = MartDesign.model_validate(state.get("mart_design") or {})
        scoped_tables = list(dict.fromkeys([*selected_tables, *(_normalize_name(table) for table in design.source_tables)]))

    filters = [str(value) for value in plan.get("filters") or []]
    references = _reference_values(plan, design)
    projected_schema, referenced_columns, schema_fallback, schema_reason = _project_schema(
        str(state.get("schema_text") or ""), scoped_tables, selected_tables, references, filters
    )
    integrity_failures, integrity_fallback, integrity_reason = _project_integrity(
        str(state.get("integrity_text") or ""), selected_tables, referenced_columns
    )
    common: dict[str, Any] = {
        "user_question": str(state.get("user_question") or ""),
        "filters": filters,
        "selected_tables": selected_tables,
        "required_columns": [str(value) for value in plan.get("required_columns") or []],
        "business_keys": {str(key): str(value) for key, value in (plan.get("business_keys") or {}).items()},
        "previous_feedback": previous_feedback,
        "schema": projected_schema,
        "integrity_failures": integrity_failures,
    }
    if route_kind == "comprehensive" and design is not None:
        context: SQLGenerationContext = ComprehensiveSQLGenerationContext(
            **common,
            target_table=f"{ALLOWED_MART_SCHEMA}.{_normalize_name(design.mart_name)}",
            source_grains=design.source_grains,
            final_grain={
                "description": design.grain,
                "grain_columns": design.grain_columns,
                "deduplication_keys": design.deduplication_keys,
            },
            column_plan=[
                column.model_dump(exclude={"inclusion_reason"}) for column in design.column_plan
            ],
            metric_support=[metric.model_dump() for metric in design.metric_support],
            aggregation_policy=design.aggregation_policy,
        )
    else:
        context = SimpleSQLGenerationContext(
            **common,
            dimensions=[str(value) for value in plan.get("dimensions") or []],
            required_aggregations=[str(value) for value in plan.get("required_aggregations") or []],
            expected_result_shape=(plan.get("validation_contract") or {}).get("expected_result_shape"),
        )
    context_json = context.model_dump_json(exclude_none=True, by_alias=True)
    legacy_lengths = _component_lengths({
        "user_question": common["user_question"],
        "plan": json.dumps(plan, ensure_ascii=False, indent=2),
        "mart_design": json.dumps(state.get("mart_design") or {}, ensure_ascii=False, indent=2),
        "schema": str(state.get("schema_text") or ""),
        "integrity": str(state.get("integrity_text") or ""),
        "feedback": previous_feedback or "없음",
    })
    projected_lengths = _component_lengths(context.model_dump(exclude_none=True, by_alias=True))
    diagnostics = {
        "retry_count": int(state.get("retry_count") or 0),
        "path": route_kind,
        "fallback": schema_fallback or integrity_fallback,
        "fallback_components": [
            component for component, used in (("schema", schema_fallback), ("integrity", integrity_fallback)) if used
        ],
        "fallback_reason_codes": {
            key: value for key, value in (("schema", schema_reason), ("integrity", integrity_reason)) if value
        },
        "legacy_component_chars": legacy_lengths,
        "projected_component_chars": projected_lengths,
        "legacy_total_chars": sum(legacy_lengths.values()),
        "projected_total_chars": len(context_json),
    }
    return GenerationContextResult(context=context, context_json=context_json, diagnostics=diagnostics)

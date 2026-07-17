from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp


AGGREGATION_ALIASES = {
    "COUNT_DISTINCT": "COUNT_DISTINCT",
    "COUNTDISTINCT": "COUNT_DISTINCT",
    "DISTINCT_COUNT": "COUNT_DISTINCT",
    "COUNT DISTINCT": "COUNT_DISTINCT",
    "DEDUP": "DEDUPLICATE",
    "DEDUPLICATE": "DEDUPLICATE",
    "ROW_NUMBER": "DEDUPLICATE",
}


@dataclass(frozen=True)
class SQLFeatures:
    source_tables: set[str] = field(default_factory=set)
    cte_names: set[str] = field(default_factory=set)
    target_tables: set[str] = field(default_factory=set)
    aggregations: set[str] = field(default_factory=set)
    window_functions: set[str] = field(default_factory=set)
    top_level_aggregations: set[str] = field(default_factory=set)
    top_level_group_by: bool = False
    parse_errors: list[str] = field(default_factory=list)


def canonical_identifier(value: Any) -> str:
    return str(value or "").replace("`", "").strip().lower()


def bare_identifier(value: Any) -> str:
    return canonical_identifier(value).split(".")[-1]


def canonical_aggregation(value: Any) -> str:
    raw = str(value or "").strip()
    parsed = _canonical_aggregation_expression(raw)
    if parsed:
        return parsed
    token = "".join(
        ch if ch.isalnum() else "_"
        for ch in raw.upper()
    ).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    if not token:
        return ""
    return AGGREGATION_ALIASES.get(token, token)


def _canonical_aggregation_expression(value: str) -> str:
    if not value or "(" not in value:
        return ""
    try:
        expression = sqlglot.parse_one(f"SELECT {value}", read="mysql")
    except Exception:
        return ""
    features = _aggregation_features(expression)
    if "COUNT_DISTINCT" in features:
        return "COUNT_DISTINCT"
    for name in ("SUM", "AVG", "MIN", "MAX", "COUNT", "NTILE", "DEDUPLICATE"):
        if name in features:
            return name
    return ""


def extract_sql_features(sql: str, *, dialect: str = "mysql") -> SQLFeatures:
    statements = sqlglot.parse(sql or "", read=dialect)
    combined = SQLFeatures()
    source_tables: set[str] = set()
    cte_names: set[str] = set()
    target_tables: set[str] = set()
    aggregations: set[str] = set()
    window_functions: set[str] = set()
    top_level_aggregations: set[str] = set()
    top_level_group_by = False

    for statement in statements:
        if statement is None:
            continue
        statement_features = _features_for_statement(statement)
        source_tables.update(statement_features.source_tables)
        cte_names.update(statement_features.cte_names)
        target_tables.update(statement_features.target_tables)
        aggregations.update(statement_features.aggregations)
        window_functions.update(statement_features.window_functions)
        top_level_aggregations.update(statement_features.top_level_aggregations)
        top_level_group_by = top_level_group_by or statement_features.top_level_group_by

    source_tables = {
        table
        for table in source_tables
        if bare_identifier(table) not in cte_names
        and bare_identifier(table) not in {bare_identifier(target) for target in target_tables}
    }
    return SQLFeatures(
        source_tables=source_tables,
        cte_names=cte_names,
        target_tables=target_tables,
        aggregations=aggregations,
        window_functions=window_functions,
        top_level_aggregations=top_level_aggregations,
        top_level_group_by=top_level_group_by,
        parse_errors=list(combined.parse_errors),
    )


def _features_for_statement(statement: exp.Expression) -> SQLFeatures:
    cte_names = {canonical_identifier(cte.alias_or_name) for cte in statement.find_all(exp.CTE)}
    target_tables = _target_tables(statement)
    top_level_query = _top_level_query(statement)
    source_tables = {
        _table_name(table)
        for table in statement.find_all(exp.Table)
        if _table_name(table)
    }
    aggregations = _aggregation_features(statement)
    window_functions = _window_features(statement)
    top_level_aggregations = _aggregation_features(top_level_query, include_nested_queries=False)
    top_level_group_by = top_level_query.args.get("group") is not None
    return SQLFeatures(
        source_tables=source_tables,
        cte_names=cte_names,
        target_tables=target_tables,
        aggregations=aggregations,
        window_functions=window_functions,
        top_level_aggregations=top_level_aggregations,
        top_level_group_by=top_level_group_by,
    )


def _target_tables(statement: exp.Expression) -> set[str]:
    targets: set[str] = set()
    if isinstance(statement, exp.Create) and isinstance(statement.this, exp.Table):
        targets.add(_table_name(statement.this))
    if isinstance(statement, exp.Insert) and isinstance(statement.this, exp.Table):
        targets.add(_table_name(statement.this))
    return {target for target in targets if target}


def _top_level_query(statement: exp.Expression) -> exp.Expression:
    if isinstance(statement, exp.Create) and statement.expression is not None:
        return statement.expression
    if isinstance(statement, exp.Insert) and statement.expression is not None:
        return statement.expression
    return statement


def _table_name(table: exp.Table) -> str:
    name = canonical_identifier(table.name)
    db = canonical_identifier(table.db)
    if db:
        return f"{db}.{name}"
    return name


def _aggregation_features(statement: exp.Expression, *, include_nested_queries: bool = True) -> set[str]:
    features: set[str] = set()
    for aggregate in _iter_aggregates(statement, include_nested_queries=include_nested_queries):
        name = aggregate.key.upper()
        if isinstance(aggregate, exp.Count):
            features.add("COUNT_DISTINCT" if isinstance(aggregate.this, exp.Distinct) else "COUNT")
            features.add("COUNT")
        elif isinstance(aggregate, exp.Sum):
            features.add("SUM")
        elif isinstance(aggregate, exp.Avg):
            features.add("AVG")
        elif isinstance(aggregate, exp.Min):
            features.add("MIN")
        elif isinstance(aggregate, exp.Max):
            features.add("MAX")
        elif isinstance(aggregate, exp.Ntile):
            features.add("NTILE")
        else:
            features.add(canonical_aggregation(name))
    selects = statement.find_all(exp.Select) if include_nested_queries else [statement] if isinstance(statement, exp.Select) else []
    if any(isinstance(select, exp.Select) and select.args.get("distinct") for select in selects):
        features.add("DEDUPLICATE")
    windows = statement.find_all(exp.Window) if include_nested_queries else _iter_top_level_windows(statement)
    if any(isinstance(window.this, exp.RowNumber) for window in windows):
        features.add("DEDUPLICATE")
    return {feature for feature in features if feature}


def _iter_aggregates(statement: exp.Expression, *, include_nested_queries: bool) -> list[exp.AggFunc]:
    if include_nested_queries:
        return list(statement.find_all(exp.AggFunc))
    nested_queries = {
        id(query)
        for query in statement.find_all(exp.Query)
        if query is not statement
    }
    aggregates: list[exp.AggFunc] = []
    for aggregate in statement.find_all(exp.AggFunc):
        if not _has_parent_in(aggregate, nested_queries):
            aggregates.append(aggregate)
    return aggregates


def _iter_top_level_windows(statement: exp.Expression) -> list[exp.Window]:
    nested_queries = {
        id(query)
        for query in statement.find_all(exp.Query)
        if query is not statement
    }
    windows: list[exp.Window] = []
    for window in statement.find_all(exp.Window):
        if not _has_parent_in(window, nested_queries):
            windows.append(window)
    return windows


def _has_parent_in(node: exp.Expression, parent_ids: set[int]) -> bool:
    parent = node.parent
    while parent is not None:
        if id(parent) in parent_ids:
            return True
        parent = parent.parent
    return False


def _window_features(statement: exp.Expression) -> set[str]:
    features: set[str] = set()
    for window in statement.find_all(exp.Window):
        this = window.this
        if this is not None:
            features.add(canonical_aggregation(this.key.upper()))
    return {feature for feature in features if feature}

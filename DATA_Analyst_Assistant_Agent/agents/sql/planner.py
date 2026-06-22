"""SQL plan generation for the SQL-Agent.

Deterministic keyword-based plan with route-aware SQL templates.
Structured so real schema/catalog lookup or LLM-based generation can be inserted later.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request

from pydantic import BaseModel, Field

from DATA_Analyst_Assistant_Agent.shared.llm import DEFAULT_OPENROUTER_MODEL, OPENROUTER_BASE_URL, get_model_name
from DATA_Analyst_Assistant_Agent.shared.contracts import AnalysisPlan


class SQLPlan(BaseModel):
    """Structured SQL plan produced by the planner."""
    selected_tables: list[str] = Field(default_factory=list)
    selected_columns: list[str] = Field(default_factory=list)
    metric: str | None = None
    dimension: str | None = None
    filters: list[str] = Field(default_factory=list)
    generated_sql: str = "SELECT 1 AS sample_value"
    reasoning: str = ""
    route_kind: str = "simple"
    planner_mode: str = "deterministic"
    confidence: float | None = None


def _detect_metric(query: str) -> str | None:
    query_lower = query.lower()
    if any(kw in query for kw in ("매출", "수익")) or any(kw in query_lower for kw in ("revenue", "sales")):
        return "revenue"
    if any(kw in query for kw in ("주문", "건수")) or "order" in query_lower:
        return "order_count"
    return None


def _detect_dimension(query: str) -> str | None:
    query_lower = query.lower()
    if any(kw in query for kw in ("월별", "추이")) or any(kw in query_lower for kw in ("monthly", "trend")):
        return "month"
    if any(kw in query for kw in ("일별",)) or "daily" in query_lower:
        return "day"
    return None


def build_sql_plan(user_query: str, analysis_plan: AnalysisPlan | None = None) -> SQLPlan:
    """Build a SQL plan from the user query and optional analysis plan."""
    requested_mode = analysis_plan.planner_mode if analysis_plan else "deterministic"
    if requested_mode == "llm":
        llm_plan = _try_build_llm_sql_plan(user_query, analysis_plan)
        if llm_plan is not None:
            return llm_plan
    return _build_deterministic_sql_plan(user_query, analysis_plan)


def _build_deterministic_sql_plan(user_query: str, analysis_plan: AnalysisPlan | None = None) -> SQLPlan:
    """Build a deterministic SQL plan from the user query and optional analysis plan."""
    metric = _detect_metric(user_query)
    dimension = _detect_dimension(user_query)
    route_kind = analysis_plan.route_kind if analysis_plan else "simple"

    # Use analysis_plan hints if available
    if analysis_plan:
        metric = metric or analysis_plan.metric
        dimension = dimension or analysis_plan.dimension

    sql = _build_template_sql(route_kind, metric, dimension)
    selected_columns = _infer_columns(metric, dimension, route_kind)

    return SQLPlan(
        selected_tables=["source_table"],
        selected_columns=selected_columns,
        metric=metric,
        dimension=dimension,
        filters=[],
        generated_sql=sql,
        reasoning=_build_reasoning(route_kind, metric, dimension),
        route_kind=route_kind,
        planner_mode="deterministic",
    )


def _try_build_llm_sql_plan(user_query: str, analysis_plan: AnalysisPlan | None) -> SQLPlan | None:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or analysis_plan is None:
        return None
    prompt = _build_llm_system_prompt(
        catalog_summary=analysis_plan.catalog_summary,
        retry_context=analysis_plan.retry_context,
    )
    try:
        raw_text = _call_llm_planner(api_key=api_key, system_prompt=prompt, user_query=user_query)
        return _parse_llm_sql_plan(raw_text, analysis_plan)
    except (ValueError, OSError, error.URLError):
        return None


def _build_llm_system_prompt(
    *,
    catalog_summary: dict[str, Any] | None,
    retry_context: dict[str, Any] | None,
) -> str:
    catalog_json = json.dumps(_sanitize_catalog_summary(catalog_summary), ensure_ascii=False, indent=2)
    retry_json = json.dumps(_coerce_retry_context(retry_context), ensure_ascii=False, indent=2)
    return (
        "You generate safe analytical SQL for MySQL.\n"
        "Allowed output: exactly one SELECT statement or one WITH statement.\n"
        "Forbidden: INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, CALL, and any multi statement SQL.\n"
        "Do not use DuckDB-specific or SQLite-specific functions or syntax.\n"
        "Return JSON only.\n"
        "<MYSQL_CATALOG_JSON>\n"
        f"{catalog_json}\n"
        "</MYSQL_CATALOG_JSON>\n"
        "<PREVIOUS_ERROR>\n"
        f"{retry_json}\n"
        "</PREVIOUS_ERROR>"
    )


def _sanitize_catalog_summary(catalog_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not catalog_summary:
        return {"tables": {}}
    tables = catalog_summary.get("tables", {})
    sanitized_tables: dict[str, Any] = {}
    for table_name, table_info in tables.items():
        columns = (table_info or {}).get("columns", {})
        sanitized_tables[table_name] = {
            "columns": {
                column_name: {
                    "type": (column_info or {}).get("type", ""),
                    "nullable": bool((column_info or {}).get("nullable", True)),
                }
                for column_name, column_info in columns.items()
            }
        }
    return {"tables": sanitized_tables}


def _coerce_retry_context(retry_context: dict[str, Any] | None) -> dict[str, Any]:
    if not retry_context:
        return {"message": ""}
    return {
        "code": retry_context.get("code", ""),
        "message": retry_context.get("message", ""),
        "query": retry_context.get("query", ""),
        "datasource_id": retry_context.get("datasource_id", ""),
        "step": retry_context.get("step", ""),
        "hint": retry_context.get("hint", ""),
    }


def _call_llm_planner(*, api_key: str, system_prompt: str, user_query: str) -> str:
    if os.getenv("OPENROUTER_API_KEY"):
        return _call_openrouter_planner(api_key=api_key, system_prompt=system_prompt, user_query=user_query)
    return _call_gemini_planner(api_key=api_key, system_prompt=system_prompt, user_query=user_query)


def _call_openrouter_planner(*, api_key: str, system_prompt: str, user_query: str) -> str:
    endpoint = os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL).rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if os.getenv("OPENROUTER_HTTP_REFERER"):
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER", "")
    if os.getenv("OPENROUTER_APP_TITLE"):
        headers["X-OpenRouter-Title"] = os.getenv("OPENROUTER_APP_TITLE", "")
    payload = {
        "model": get_model_name(default=DEFAULT_OPENROUTER_MODEL),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with request.urlopen(req, timeout=20) as response:
        body = json.loads(response.read().decode("utf-8"))
    return _extract_openrouter_response_text(body)


def _call_gemini_planner(*, api_key: str, system_prompt: str, user_query: str) -> str:
    model_name = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_query}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
    }
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with request.urlopen(req, timeout=20) as response:
        body = json.loads(response.read().decode("utf-8"))
    return _extract_gemini_response_text(body)


def _extract_openrouter_response_text(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("LLM response missing choices.")
    message = (choices[0] or {}).get("message") or {}
    text = message.get("content") or ""
    if not str(text).strip():
        raise ValueError("LLM response missing text payload.")
    return str(text)


def _extract_gemini_response_text(body: dict[str, Any]) -> str:
    candidates = body.get("candidates") or []
    if not candidates:
        raise ValueError("LLM response missing candidates.")
    content = (candidates[0] or {}).get("content") or {}
    parts = content.get("parts") or []
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    if not text.strip():
        raise ValueError("LLM response missing text payload.")
    return text


def _parse_llm_sql_plan(raw_text: str, analysis_plan: AnalysisPlan) -> SQLPlan:
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("LLM planner payload must be a JSON object.")
    generated_sql = payload.get("generated_sql")
    if not isinstance(generated_sql, str) or not generated_sql.strip():
        raise ValueError("LLM planner payload missing generated_sql.")
    return SQLPlan(
        selected_tables=[str(x) for x in payload.get("selected_tables", [])],
        selected_columns=[str(x) for x in payload.get("selected_columns", [])],
        metric=analysis_plan.metric,
        dimension=analysis_plan.dimension,
        filters=list(analysis_plan.filters),
        generated_sql=generated_sql.strip(),
        reasoning=str(payload.get("reasoning") or "LLM plan selected SQL from catalog-aware prompt."),
        route_kind=analysis_plan.route_kind,
        planner_mode="llm",
        confidence=_coerce_confidence(payload.get("confidence")),
    )


def _coerce_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ── Route-aware SQL templates ──

_TEMPLATES: dict[str, str] = {
    "simple_metric": "SELECT 1 AS {metric}",
    "simple_dimension_metric": "SELECT 1 AS {dimension}, 1 AS {metric}",
    "trend": (
        "WITH monthly_data AS ("
        "  SELECT 1 AS {dimension}, 1 AS {metric}"
        ") "
        "SELECT {dimension}, {metric} FROM monthly_data ORDER BY {dimension}"
    ),
    "eda": (
        "SELECT 1 AS column_name, 1 AS non_null_count, 1 AS distinct_count, "
        "1 AS sample_value"
    ),
    "comprehensive": (
        "WITH base_data AS ("
        "  SELECT 1 AS {dimension}, 1 AS {metric}"
        ") "
        "SELECT {dimension}, {metric} FROM base_data ORDER BY {dimension}"
    ),
    "fallback": "SELECT 1 AS sample_value",
}


def _build_template_sql(route_kind: str, metric: str | None, dimension: str | None) -> str:
    """Select and fill a SQL template based on route_kind and detected fields.

    MVP: all templates produce self-contained SELECTs that need no real tables.
    Extension point: replace with real schema-aware SQL generation.
    """
    m = metric or "value"
    d = dimension or "category"

    if route_kind == "trend":
        return _TEMPLATES["trend"].format(metric=m, dimension=d)
    if route_kind == "eda":
        return _TEMPLATES["eda"]
    if route_kind == "comprehensive":
        return _TEMPLATES["comprehensive"].format(metric=m, dimension=d)
    # simple / mart / unknown
    if metric and dimension:
        return _TEMPLATES["simple_dimension_metric"].format(metric=m, dimension=d)
    if metric:
        return _TEMPLATES["simple_metric"].format(metric=m)
    return _TEMPLATES["fallback"]


def _infer_columns(metric: str | None, dimension: str | None, route_kind: str) -> list[str]:
    if route_kind == "eda":
        return ["column_name", "non_null_count", "distinct_count", "sample_value"]
    cols: list[str] = []
    if dimension:
        cols.append(dimension)
    if metric:
        cols.append(metric)
    return cols or ["sample_value"]


def _build_reasoning(route_kind: str, metric: str | None, dimension: str | None) -> str:
    parts = [f"route={route_kind}"]
    if metric:
        parts.append(f"metric={metric}")
    if dimension:
        parts.append(f"dimension={dimension}")
    return f"Deterministic plan: {', '.join(parts)}."

from __future__ import annotations

from typing import Any

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisContext,
    AnalysisSelectionResponse,
    ReviewRequest,
)
from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState


def build_analysis_context(
    state: OrchestrationState,
    dataframe: pd.DataFrame,
    eda_profiles: list[dict[str, Any]],
    *,
    question_type: str | None = None,
    sample_limit: int = 5,
    selection_response: AnalysisSelectionResponse | None = None,
    review_request: ReviewRequest | None = None,
) -> AnalysisContext:
    """Expose only task-relevant data and a small sample to an optional LLM planner."""

    numeric = list(dataframe.select_dtypes(include="number").columns)
    categorical = [column for column in dataframe.columns if column not in numeric]
    temporal = [
        column
        for column in dataframe.columns
        if pd.api.types.is_datetime64_any_dtype(dataframe[column])
        or any(token in column.casefold() for token in ("date", "time", "month", "year"))
    ]
    samples = []
    if not dataframe.empty:
        samples = dataframe.head(sample_limit).where(pd.notna(dataframe), None).to_dict(orient="records")
    column_profiles = _column_profiles(dataframe, numeric)

    quality_statuses: list[str] = []
    issues: list[str] = []
    candidate_insights: list[str] = []
    candidate_hypotheses: list[str] = []
    for profile in eda_profiles:
        profile_block = profile.get("profile", profile)
        status = profile_block.get("quality_status")
        if status:
            quality_statuses.append(str(status))
        issues.extend(str(item) for item in profile_block.get("key_issues", []) or [])
        candidate_insights.extend(_candidate_texts(profile.get("insight_result")))
        candidate_hypotheses.extend(_candidate_texts(profile.get("hypotheses")))
        for caution in profile.get("cautions", []) or []:
            if isinstance(caution, dict) and caution.get("message_ko"):
                issues.append(str(caution["message_ko"]))

    plan = state.plan
    # 상류 SQL 원천 테이블의 GE 정합성 이슈를 스코핑해 코드생성/검증이 참고하게 한다(#130).
    # #123 함수 재사용(fail_only+테이블스코핑+100줄 캡). source_tables 없으면 빈 리스트로 폴백.
    known_data_quality_issues: list[str] = []
    plan_source_tables = list(plan.source_tables) if plan and plan.source_tables else []
    if plan_source_tables:
        from DATA_Analyst_Assistant_Agent.agents.sql.validator.integrity_loader import load_scoped_integrity_text
        integrity_text = load_scoped_integrity_text(plan_source_tables)
        known_data_quality_issues = [
            line.lstrip("- ").strip()
            for line in integrity_text.splitlines()
            if line.strip().startswith("- ")
        ]
    retry_context = state.retry_context or {}
    last_failure_payload = retry_context.get("last_failure")
    last_failure = None
    if isinstance(last_failure_payload, dict):
        last_failure = {
            "reason_code": str(last_failure_payload.get("reason_code") or "none"),
            "failure_reason": str(last_failure_payload.get("failure_reason") or ""),
        }
    elif isinstance((retry_context.get("agent_feedback") or {}).get("analysis_agent"), dict):
        # 하드 실패(failure_streaks) 기록이 없을 때만 semantic 검증 피드백으로 폴백한다
        # (하드 실패가 더 구체적이므로 우선). missing_evidence를 reason_code 없이 텍스트로 합친다.
        feedback = retry_context["agent_feedback"]["analysis_agent"]
        reason = str(feedback.get("reason") or "")
        missing = feedback.get("missing_evidence") or []
        missing_text = f" 누락된 근거: {', '.join(str(item) for item in missing)}." if missing else ""
        last_failure = {
            "reason_code": "semantic_validation_failed",
            "failure_reason": f"{reason}{missing_text}".strip(),
        }
    analysis_contract, mart_columns, contract_issues = _analysis_data_contract_from_plan(plan)
    return AnalysisContext(
        user_question=state.user_query,
        goal=state.goal or (plan.goal if plan else state.user_query),
        route_kind=state.route_kind or (plan.route_kind if plan else "simple"),
        question_type=question_type or _question_type_from_state(state),
        metric_hint=plan.metric if plan else None,
        dimension_hint=plan.dimension if plan else None,
        row_count=len(dataframe),
        columns=list(dataframe.columns),
        numeric_columns=numeric,
        categorical_columns=categorical,
        temporal_columns=temporal,
        column_profiles=column_profiles,
        sample_rows=samples,
        eda_quality_statuses=quality_statuses,
        eda_key_issues=list(dict.fromkeys(issues)),
        eda_candidate_insights=list(dict.fromkeys(candidate_insights)),
        eda_candidate_hypotheses=list(dict.fromkeys(candidate_hypotheses)),
        known_data_quality_issues=known_data_quality_issues,
        analysis_data_contract=analysis_contract,
        mart_columns=mart_columns,
        contract_issues=contract_issues,
        source_artifact_ids=[item for ids in state.artifact_ids.values() for item in ids],
        last_failure=last_failure,
        review_request=review_request,
        selection_response=selection_response,
    )


def _question_type_from_state(state: OrchestrationState) -> str | None:
    """Forward-compatible hook for a supervisor-owned question type contract."""

    direct = getattr(state, "question_type", None)
    if direct:
        return str(direct)
    if state.plan:
        planned = getattr(state.plan, "question_type", None)
        if planned:
            return str(planned)
    for source in (state.retry_context or {}, state.error_state or {}):
        value = source.get("question_type")
        if value:
            return str(value)
    return None


def _candidate_texts(value: Any, *, max_items: int = 12, max_chars: int = 500) -> list[str]:
    """Normalize exploratory EDA text into bounded candidate hints."""

    items: list[str] = []
    _collect_candidate_texts(value, items)
    normalized: list[str] = []
    for item in items:
        text = _clean_candidate_text(item, max_chars=max_chars)
        if text:
            normalized.append(text)
        if len(normalized) >= max_items:
            break
    return normalized


def _collect_candidate_texts(value: Any, items: list[str]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        for line in value.splitlines():
            line = line.strip()
            if line:
                items.append(line)
        return
    if isinstance(value, dict):
        preferred_keys = (
            "hypothesis",
            "insight",
            "summary",
            "finding",
            "rationale",
            "description",
            "text",
        )
        found = False
        for key in preferred_keys:
            if key in value:
                _collect_candidate_texts(value[key], items)
                found = True
        if not found:
            for nested in value.values():
                _collect_candidate_texts(nested, items)
        return
    if isinstance(value, (list, tuple, set)):
        for nested in value:
            _collect_candidate_texts(nested, items)


def _clean_candidate_text(text: str, *, max_chars: int) -> str:
    cleaned = " ".join(str(text).split())
    cleaned = cleaned.lstrip("-*0123456789. )\t")
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 3].rstrip() + "..."
    return cleaned


def _analysis_data_contract_from_plan(plan: Any) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    if plan is None:
        return {}, [], []
    contract = dict(getattr(plan, "analysis_data_contract", None) or {})
    mart_design = dict(getattr(plan, "mart_design", None) or {})
    generated_sql = str(getattr(plan, "generated_sql", "") or getattr(plan, "source_sql", "") or "")
    if not contract:
        contract = _contract_from_mart_design(mart_design, plan, generated_sql)
    if not contract:
        contract = _contract_from_plan_sql(plan, generated_sql)

    mart_columns = [item for item in list(contract.get("derived_columns") or []) if isinstance(item, dict)]
    inferred = _infer_column_lineage_from_sql(generated_sql)
    known = {str(item.get("output_column") or "") for item in mart_columns}
    for item in inferred:
        if item["output_column"] not in known:
            mart_columns.append(item)
    if mart_columns and not contract.get("derived_columns"):
        contract["derived_columns"] = mart_columns

    issues: list[str] = []
    if getattr(plan, "target_table", None) and not str(contract.get("row_grain") or "").strip():
        issues.append("SQL 데이터마트의 row_grain이 비어 있습니다.")
    if any(item.get("output_column") == "month" for item in mart_columns):
        has_month_source = any(
            item.get("output_column") == "month" and item.get("source_columns")
            for item in mart_columns
        )
        if not has_month_source:
            issues.append("month 컬럼의 원본/파생 근거가 부족하므로 추세 해석을 제한해야 합니다.")
    return contract, mart_columns, issues


def _contract_from_mart_design(mart_design: dict[str, Any], plan: Any, generated_sql: str) -> dict[str, Any]:
    if not mart_design:
        return {}
    column_plan = [item for item in mart_design.get("column_plan") or [] if isinstance(item, dict)]
    grain_columns = list(mart_design.get("grain_columns") or [])
    return {
        "target_table": getattr(plan, "target_table", None) or mart_design.get("mart_name"),
        "row_grain": mart_design.get("grain") or getattr(plan, "business_grain", None) or "",
        "grain_columns": grain_columns,
        "entity_keys": [column for column in grain_columns if not _looks_temporal_column(column)],
        "time_basis": [
            item for item in column_plan
            if _looks_temporal_column(str(item.get("output_column") or ""))
            or any(_looks_temporal_column(str(source)) for source in item.get("source_columns") or [])
        ],
        "derived_columns": column_plan,
        "aggregation_rules": [
            item for item in column_plan
            if str(item.get("aggregation_method") or "none").lower() != "none"
        ],
        "deduplication_keys": list(mart_design.get("deduplication_keys") or []),
        "source_tables": list(mart_design.get("source_tables") or getattr(plan, "source_tables", []) or []),
        "source_grains": dict(mart_design.get("source_grains") or {}),
        "safe_interpretations": [
            "선언된 row_grain 기준에서만 데이터마트를 해석합니다.",
            "집계 수준의 연관성을 개별 주문/사용자 수준 인과로 해석하지 않습니다.",
        ],
        "generated_sql": generated_sql,
    }


def _contract_from_plan_sql(plan: Any, generated_sql: str) -> dict[str, Any]:
    if not generated_sql and not getattr(plan, "target_table", None):
        return {}
    inferred = _infer_column_lineage_from_sql(generated_sql)
    group_columns = _infer_group_columns_from_sql(generated_sql)
    return {
        "target_table": getattr(plan, "target_table", None),
        "row_grain": getattr(plan, "business_grain", None) or ", ".join(group_columns),
        "grain_columns": group_columns,
        "entity_keys": [column for column in group_columns if not _looks_temporal_column(column)],
        "time_basis": [
            item for item in inferred
            if _looks_temporal_column(item.get("output_column", ""))
            or any(_looks_temporal_column(source) for source in item.get("source_columns", []))
        ],
        "derived_columns": inferred,
        "aggregation_rules": [
            item for item in inferred
            if str(item.get("aggregation_method") or "none").lower() != "none"
        ],
        "deduplication_keys": group_columns,
        "source_tables": list(getattr(plan, "source_tables", []) or []),
        "source_grains": {},
        "safe_interpretations": [
            "SQL에서 파생된 컬럼은 생성 SQL 표현식에 따라 해석합니다.",
            "판매자/고객/그룹 요약은 집계 수준 한계를 함께 명시합니다.",
        ],
        "generated_sql": generated_sql,
    }


def _infer_column_lineage_from_sql(sql: str) -> list[dict[str, Any]]:
    if not sql.strip():
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
            source_columns = sorted({column.sql(dialect="mysql") for column in expression.find_all(exp.Column)})
            aggregation = "none"
            agg = next(expression.find_all(exp.AggFunc), None)
            if agg is not None:
                aggregation = agg.key.upper()
            items.append(
                {
                    "output_column": alias,
                    "source_columns": source_columns or [alias],
                    "calculation_type": "derived" if expression.find(exp.Func) or expression.find(exp.AggFunc) else "passthrough",
                    "calculation_rule": expression.sql(dialect="mysql"),
                    "aggregation_method": aggregation,
                }
            )
        return items
    except Exception:
        return []


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


def _looks_temporal_column(name: str) -> bool:
    normalized = str(name or "").casefold()
    return any(token in normalized for token in ("date", "time", "month", "year", "timestamp"))


def _column_profiles(dataframe: pd.DataFrame, numeric_columns: list[str]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for column in dataframe.columns:
        series = dataframe[column]
        profile: dict[str, Any] = {
            "dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
        }
        if column in numeric_columns:
            values = pd.to_numeric(series, errors="coerce").dropna()
            if not values.empty:
                profile.update(
                    min=float(values.min()),
                    max=float(values.max()),
                    mean=float(values.mean()),
                    std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                )
        else:
            profile["top_values"] = {
                str(key): int(value) for key, value in series.value_counts(dropna=False).head(5).items()
            }
        profiles[column] = profile
    return profiles

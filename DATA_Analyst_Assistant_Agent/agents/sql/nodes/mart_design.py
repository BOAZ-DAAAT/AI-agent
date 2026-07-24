"""design_mart 노드: comprehensive 경로의 데이터마트 설계."""

from __future__ import annotations

import json
import re
from typing import Any

from DATA_Analyst_Assistant_Agent.agents.sql import prompts
from DATA_Analyst_Assistant_Agent.agents.sql.planner_support import (
    normalize_mart_column_lists,
    require_route_kind,
    try_llm_json,
)
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState, MartDesign


def _normalize_column_list(values: Any) -> list[str]:
    if not values:
        return []
    if not isinstance(values, list):
        values = [values]

    normalized: list[str] = []
    for item in values:
        if isinstance(item, str):
            candidate = item.strip()
        elif isinstance(item, dict):
            candidate = str(
                item.get("column_name")
                or item.get("name")
                or item.get("column")
                or item.get("field")
                or ""
            ).strip()
        else:
            candidate = str(item).strip()
        if candidate and candidate not in normalized:
            normalized.append(candidate)
    return normalized


def _normalize_mart_design_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    for key in ("source_tables", "grain_columns", "deduplication_keys"):
        normalized[key] = _normalize_column_list(normalized.get(key))
    source_grains = normalized.get("source_grains")
    if isinstance(source_grains, dict):
        normalized["source_grains"] = {
            str(table).strip(): _normalize_column_list(columns)
            for table, columns in source_grains.items()
            if str(table).strip()
        }
    normalized["column_plan"] = _repair_empty_source_columns(normalized.get("column_plan"))
    unimplemented = normalized.get("unimplemented_derivations")
    if isinstance(unimplemented, list):
        normalized["unimplemented_derivations"] = [
            {
                **item,
                "name": str(
                    item.get("name")
                    or item.get("preferred_name")
                    or item.get("output_column")
                    or ""
                ).strip(),
                "required_columns": _normalize_column_list(
                    item.get("required_columns") or item.get("source_columns")
                ),
            }
            for item in unimplemented
            if isinstance(item, dict)
        ]
    return normalized


def _repair_empty_source_columns(column_plan: Any) -> Any:
    if not isinstance(column_plan, list):
        return column_plan

    output_names: list[str] = []
    source_names: list[str] = []
    for item in column_plan:
        if not isinstance(item, dict):
            continue
        output = str(item.get("output_column") or "").strip()
        if output and output not in output_names:
            output_names.append(output)
        for source in _normalize_column_list(item.get("source_columns")):
            for candidate in (source, source.split(".")[-1]):
                if candidate and candidate not in source_names:
                    source_names.append(candidate)

    repaired: list[Any] = []
    for item in column_plan:
        if not isinstance(item, dict):
            repaired.append(item)
            continue
        current = dict(item)
        if _normalize_column_list(current.get("source_columns")):
            repaired.append(current)
            continue
        if str(current.get("calculation_type") or "").strip().lower() != "derived":
            repaired.append(current)
            continue

        text = " ".join(
            str(current.get(key) or "")
            for key in ("calculation_rule", "inclusion_reason", "output_column")
        )
        own_output = str(current.get("output_column") or "").strip()
        inferred: list[str] = []
        for candidate in [*output_names, *source_names]:
            if candidate == own_output:
                continue
            if _mentions_column(text, candidate) and candidate not in inferred:
                inferred.append(candidate)
        if inferred:
            current["source_columns"] = inferred
        repaired.append(current)
    return repaired


def _mentions_column(text: str, column: str) -> bool:
    needle = str(column or "").strip()
    if not needle:
        return False
    folded_text = text.casefold()
    folded = needle.casefold()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", folded):
        return re.search(rf"(?<![A-Za-z0-9_$]){re.escape(folded)}(?![A-Za-z0-9_$])", folded_text) is not None
    return folded in folded_text


def _validate_target_metric_support(design: MartDesign, plan: dict[str, Any]) -> None:
    target_metrics = list(dict.fromkeys(str(metric).strip() for metric in plan.get("target_metrics", []) if str(metric).strip()))
    supported_metrics = [item.metric_name for item in design.metric_support]
    if supported_metrics != target_metrics:
        raise ValueError(
            "metric_support는 고유 target_metrics를 같은 순서로 정확히 한 번 포함해야 합니다: "
            f"expected={target_metrics}, actual={supported_metrics}"
        )


def _validate_required_derivation_coverage(
    design: MartDesign,
    required_derivations: list[dict[str, Any]],
) -> None:
    expected_names = [
        str(item.get("preferred_name") or item.get("name") or "").strip()
        for item in required_derivations
    ]
    if any(not name for name in expected_names):
        raise ValueError("required_derivations의 name 또는 preferred_name이 비어 있습니다")
    if len(set(expected_names)) != len(expected_names):
        raise ValueError("required_derivations의 preferred_name은 중복될 수 없습니다")

    implemented_names = {item.output_column for item in design.column_plan}
    unimplemented_names = {item.name for item in design.unimplemented_derivations}
    missing = [
        name
        for name in expected_names
        if name not in implemented_names and name not in unimplemented_names
    ]
    duplicated = [
        name
        for name in expected_names
        if name in implemented_names and name in unimplemented_names
    ]
    if missing or duplicated:
        raise ValueError(
            "required_derivations는 column_plan.output_column 또는 "
            "unimplemented_derivations 중 정확히 한 곳에 있어야 합니다: "
            f"missing={missing}, duplicated={duplicated}"
        )


def validate_mart_design_state(
    payload: dict[str, Any],
    plan: dict[str, Any],
    required_derivations: list[dict[str, Any]] | None = None,
) -> MartDesign:
    """저장되었거나 새로 생성된 마트 설계가 현재 계약과 계획을 만족하는지 확인한다."""
    design = MartDesign(**_normalize_mart_design_payload(payload))
    _validate_target_metric_support(design, plan)
    _validate_required_derivation_coverage(design, list(required_derivations or []))
    return design


def _mart_design_validation_detail(normalized: dict[str, Any], exc: Exception) -> str:
    detail = f"LLM mart design response does not satisfy the required contract: {exc}"
    empty_source_outputs = [
        str(item.get("output_column") or "").strip()
        for item in normalized.get("column_plan") or []
        if isinstance(item, dict) and not _normalize_column_list(item.get("source_columns"))
    ]
    if empty_source_outputs:
        detail += (
            " source_columns가 빈 파생 컬럼은 계산에 사용한 원천/중간 컬럼을 명시해야 합니다. "
            f"대상 컬럼: {empty_source_outputs}."
        )
    return detail


def _mart_design_failure(*, reason_code: str, detail: str, retryable: bool) -> dict[str, Any]:
    suggested_action = "redesign_mart" if retryable else "stop_and_surface_error"
    finding = {
        "category": "sql_mart_design_failed",
        "severity": "error",
        "retryable": retryable,
        "detail": detail,
        "message": detail,
        "source": "sql_mart_designer",
        "code": reason_code,
        "suggested_action": suggested_action,
        "details": {"reason_code": reason_code},
    }
    retry_hint = {
        "retryable": retryable,
        "suggested_action": suggested_action,
        "reason_code": "sql_mart_design_failed",
        "details": {
            "mart_design_reason_code": reason_code,
            "messages": [detail],
        },
    }
    return {
        "mart_design": {},
        "validation": {
            "result": "invalid",
            "reason": detail,
            "feedback": detail,
            "findings": [finding],
            "retry_hint": retry_hint,
        },
        "validation_findings": [finding],
        "retry_hint": retry_hint,
        "feedback": detail,
        "error": detail,
    }


def design_mart(state: AgentState):
    if require_route_kind(state["plan"]) != "comprehensive":
        return {"mart_design": {}}

    response = try_llm_json(prompts.mart_design_prompt(state))
    if not response:
        return _mart_design_failure(
            reason_code="llm_empty_response",
            detail="LLM이 mart 설계 초안을 반환하지 못했습니다.",
            retryable=True,
        )

    cleaned = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return _mart_design_failure(
            reason_code="llm_json_parse_failed",
            detail="LLM mart 설계 응답을 JSON으로 파싱하지 못했습니다.",
            retryable=True,
        )

    if not isinstance(parsed, dict):
        return _mart_design_failure(
            reason_code="llm_json_not_object",
            detail="LLM mart 설계 응답이 JSON object가 아닙니다.",
            retryable=True,
        )

    normalized = normalize_mart_column_lists(_normalize_mart_design_payload(parsed))
    try:
        design = validate_mart_design_state(
            normalized,
            state.get("plan") or {},
            list(state.get("required_derivations") or []),
        )
    except Exception as exc:
        return _mart_design_failure(
            reason_code="invalid_mart_design_payload",
            detail=_mart_design_validation_detail(normalized, exc),
            retryable=True,
        )

    return {"mart_design": design.model_dump()}

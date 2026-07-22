"""유효한 이전 SQLDraft를 최소 컨텍스트로 수정하는 repair_sql 노드."""

from __future__ import annotations

import json
import re
from typing import Any

from DATA_Analyst_Assistant_Agent.agents.sql.execution_errors import REPAIR_STRATEGY_BY_MYSQL_CODE
from DATA_Analyst_Assistant_Agent.agents.sql.planner_support import normalize_generated_sql, require_route_kind, try_llm_json
from DATA_Analyst_Assistant_Agent.agents.sql.prompts.repair import repair_sql_prompt
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState, SQLDraft


def _repair_context(state: AgentState) -> tuple[str, str]:
    error_info = state.get("execution_error_info") or {}
    strategy = str(
        error_info.get("repair_strategy")
        or state.get("repair_strategy")
        or REPAIR_STRATEGY_BY_MYSQL_CODE.get(error_info.get("error_code"))
        or "none"
    )
    classification = str(error_info.get("classification") or state.get("classification") or "unknown")
    return classification, strategy


def _normalized_sql(value: Any) -> str:
    return " ".join(str(value or "").strip().rstrip(";").split()).casefold()


def _failed_component_changed(
    previous: dict[str, Any],
    repaired: dict[str, Any],
    error_info: dict[str, Any],
) -> bool:
    component = str(error_info.get("component") or "main")
    field = {"precheck": "precheck_sql", "postcheck": "postcheck_sql"}.get(component, "sql")
    return _normalized_sql(previous.get(field)) != _normalized_sql(repaired.get(field))


def _validate_data_value_repair(
    repaired: dict[str, Any],
    error_info: dict[str, Any],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    evidence: list[str] = []
    column_name = str(error_info.get("column_name") or "").strip().strip("`")
    raw_invalid_value = error_info.get("invalid_value")
    invalid_value_known = raw_invalid_value is not None
    invalid_value = str(raw_invalid_value).strip() if invalid_value_known else ""
    sql = str(repaired.get("sql") or "")
    precheck = str(repaired.get("precheck_sql") or "")
    sql_lower = sql.casefold()
    precheck_lower = precheck.casefold()
    bare_column = column_name.split(".")[-1].casefold()

    if not column_name:
        errors.append("오류 컨텍스트에 column_name이 없습니다.")
    else:
        evidence.append(f"오류 컬럼: {column_name}")
    if not invalid_value_known:
        errors.append("오류 컨텍스트에 invalid_value가 없습니다.")
    else:
        evidence.append(f"오류 값: {invalid_value}")

    for token, label in (("case", "CASE"), ("cast", "명시적 CAST"), ("null", "NULL 처리")):
        if not re.search(rf"\b{token}\b", sql_lower):
            errors.append(f"main SQL에 {label} 정규화 증거가 없습니다.")
    if bare_column and bare_column not in sql_lower:
        errors.append("main SQL이 오류 컬럼을 정규화하지 않습니다.")
    sql_has_invalid_value = (
        invalid_value.casefold() in sql_lower
        if invalid_value
        else "''" in sql or '\"\"' in sql
    )
    if invalid_value_known and not sql_has_invalid_value:
        errors.append("main SQL이 추출된 오류 값을 명시적으로 처리하지 않습니다.")

    message = str(error_info.get("message") or "").casefold()
    datetime_error = any(token in message for token in ("date", "time", "timestamp"))
    if datetime_error and not re.search(r"\bwith\b", sql_lower):
        errors.append("datetime 정규화는 정제 CTE 안에서 수행해야 합니다.")
    if datetime_error and not re.search(r"\bcast\s*\([^)]*\bas\s+(?:date|datetime|timestamp)\b", sql_lower, re.DOTALL):
        errors.append("datetime 값에 날짜/시간 타입의 명시적 CAST가 없습니다.")

    case_matches = list(re.finditer(
        r"\bcase\b(?P<body>.*?)\bend\s+(?:as\s+)?`?(?P<alias>[a-z_][\w$]*)`?",
        sql_lower,
        re.DOTALL,
    ))
    normalization_matches = []
    for match in case_matches:
        body = match.group("body")
        body_has_value = invalid_value.casefold() in body if invalid_value else "''" in body or '\"\"' in body
        if bare_column in body and body_has_value and re.search(r"\bcast\b", body) and re.search(r"\bnull\b", body):
            normalization_matches.append(match)
    aliases = [match.group("alias") for match in normalization_matches]
    if not aliases:
        errors.append("오류 컬럼·값을 CASE + CAST + NULL로 정규화한 alias가 없습니다.")
    elif not any(len(re.findall(rf"\b{re.escape(alias)}\b", sql_lower)) >= 2 for alias in aliases):
        errors.append("정제 alias가 이후 출력·정렬·집계에서 사용되지 않습니다.")
    else:
        evidence.append("정규화 표현식과 정제 alias 사용")

    if datetime_error and aliases:
        outer_selects = list(re.finditer(r"\)\s*select\b", sql_lower))
        outer_sql = sql_lower[outer_selects[-1].start() + 1:] if outer_selects else ""
        if not outer_sql or not any(re.search(rf"\b{re.escape(alias)}\b", outer_sql) for alias in aliases):
            errors.append("datetime 정제 alias가 CTE 이후 SQL에서 사용되지 않습니다.")
        if bare_column and all(alias != bare_column for alias in aliases) and re.search(
            rf"\b{re.escape(bare_column)}\b",
            outer_sql,
        ):
            errors.append("CTE 이후 SQL이 정제 alias 대신 원본 datetime 컬럼을 사용합니다.")

    where_clauses = re.findall(
        r"\bwhere\b(.*?)(?=\b(?:group\s+by|order\s+by|having|limit|union)\b|$)",
        sql_lower,
        re.DOTALL,
    )
    for clause in where_clauses:
        clause_has_value = invalid_value.casefold() in clause if invalid_value else "''" in clause or '\"\"' in clause
        if bare_column and invalid_value_known and bare_column in clause and clause_has_value:
            errors.append("오류 값을 WHERE로 제거하는 방식은 허용되지 않습니다.")
            break

    if re.search(r"\b(?:substring|substr|left|right)\s*\(", sql_lower):
        errors.append("임의 문자열 절단 표현식은 데이터값 repair에서 허용되지 않습니다.")

    if not precheck:
        errors.append("비정상 값 건수를 확인할 precheck_sql이 없습니다.")
    else:
        if not re.search(r"\b(?:count|sum)\s*\(", precheck_lower):
            errors.append("precheck_sql이 비정상 값 건수를 집계하지 않습니다.")
        if bare_column and bare_column not in precheck_lower:
            errors.append("precheck_sql이 오류 컬럼을 검사하지 않습니다.")
        precheck_has_invalid_value = (
            invalid_value.casefold() in precheck_lower
            if invalid_value
            else "''" in precheck or '\"\"' in precheck
        )
        if invalid_value_known and not precheck_has_invalid_value:
            errors.append("precheck_sql이 추출된 오류 값을 검사하지 않습니다.")
        if not errors or all("precheck_sql" not in item for item in errors):
            evidence.append("오류 컬럼·값의 비정상 건수 precheck")
    return errors, evidence


def _validate_repair_result(
    previous: dict[str, Any],
    repaired: dict[str, Any],
    strategy: str,
    error_info: dict[str, Any],
) -> tuple[list[str], list[str]]:
    if not _failed_component_changed(previous, repaired, error_info):
        return ["repair 결과가 이전 SQL과 동일합니다."], []
    evidence = ["SQLDraft 계약 통과", "이전 SQL 대비 SQL 변경"]
    if strategy == "normalize_invalid_value":
        errors, data_evidence = _validate_data_value_repair(repaired, error_info)
        return errors, [*evidence, *data_evidence]
    if strategy in {"rewrite_syntax", "rewrite_identifier", "rewrite_aggregation"}:
        return [], evidence
    return [f"지원하지 않는 repair 전략입니다: {strategy}"], []


def _repair_failure(
    state: AgentState,
    reason_code: str,
    detail: str,
    *,
    retryable: bool = True,
    validation_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    classification, strategy = _repair_context(state)
    finding = {
        "category": "sql_generation_failed",
        "severity": "error",
        "retryable": retryable,
        "detail": detail,
        "message": detail,
        "source": "sql_repair",
        "code": reason_code,
        "suggested_action": "repair_sql" if retryable else "stop_and_surface_error",
        "details": {"reason_code": reason_code, "generation_stage": "repair"},
    }
    retry_hint = {
        "retryable": retryable,
        "suggested_action": "repair_sql" if retryable else "stop_and_surface_error",
        "reason_code": "sql_generation_failed",
        "details": {
            "generation_stage": "repair",
            "generation_reason_code": reason_code,
            "messages": [detail],
            "original_retry_hint": state.get("retry_hint") or {},
        },
    }
    previous_findings = list(state.get("validation_findings") or [])
    findings = previous_findings + [finding]
    return {
        "sql_draft": {},
        "generation_source": "failed",
        "generation_failure_reason": reason_code,
        "validation": {
            "result": "invalid",
            "reason": detail,
            "feedback": detail,
            "findings": findings,
            "retry_hint": retry_hint,
        },
        "validation_findings": findings,
        "retry_hint": retry_hint,
        "feedback": detail,
        "error": state.get("error") or detail,
        "classification": classification,
        "repair_strategy": strategy,
        "repair_attempted": True,
        "repair_validation_result": {
            "result": "failed",
            "reason_code": reason_code,
            "details": validation_details or {"messages": [detail]},
        },
    }


def repair_sql(state: AgentState) -> dict[str, Any]:
    classification, strategy = _repair_context(state)
    error_info = dict(state.get("execution_error_info") or {})
    retryable = bool(error_info.get("retryable", True))
    if not retryable or strategy == "none":
        return _repair_failure(
            state,
            "repair_strategy_not_applicable",
            "실행 오류가 자동 SQL repair 대상 전략으로 분류되지 않았습니다.",
            retryable=False,
        )

    previous_draft = state.get("previous_sql_draft") or state.get("sql_draft") or {}
    if not str(previous_draft.get("sql") or "").strip():
        return _repair_failure(
            state,
            "missing_previous_sql_draft",
            "repair할 유효한 이전 SQLDraft가 없습니다.",
            retryable=False,
        )

    response = try_llm_json(repair_sql_prompt(state, previous_draft))
    if not response or not response.strip():
        return _repair_failure(state, "llm_empty_response", "LLM이 SQL repair 초안을 반환하지 않았습니다.")
    cleaned = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return _repair_failure(state, "llm_json_parse_failed", "LLM SQL repair 응답을 JSON으로 파싱하지 못했습니다.")
    if not isinstance(parsed, dict):
        return _repair_failure(state, "llm_json_not_object", "LLM SQL repair 응답이 JSON object가 아닙니다.")

    route_kind = require_route_kind(state.get("plan") or {})
    try:
        normalized = normalize_generated_sql(
            parsed,
            route_kind,
            default_business_grain=(state.get("mart_design") or {}).get("grain") or (state.get("plan") or {}).get("grain"),
        )
        repaired_draft = SQLDraft(**normalized).model_dump()
        if not str(repaired_draft.get("sql") or "").strip().strip(";"):
            raise ValueError("repair SQL이 비어 있습니다")
    except Exception as exc:
        return _repair_failure(state, "repair_contract_invalid", f"SQL repair 응답이 SQLDraft 계약을 만족하지 않습니다: {exc}")

    validation_errors, evidence = _validate_repair_result(
        previous_draft,
        repaired_draft,
        strategy,
        error_info,
    )
    if validation_errors:
        reason_code = "repair_no_effect" if validation_errors == ["repair 결과가 이전 SQL과 동일합니다."] else "repair_strategy_validation_failed"
        return _repair_failure(
            state,
            reason_code,
            "SQL repair 로컬 검사에 실패했습니다: " + " / ".join(validation_errors),
            retryable=False,
            validation_details={"errors": validation_errors, "evidence": evidence},
        )

    return {
        "sql_draft": repaired_draft,
        "generation_source": "repair",
        "generation_failure_reason": "",
        "validation": {},
        "validation_findings": [],
        "retry_hint": {},
        "feedback": "",
        "error": "",
        "statement_results": [],
        "failed_sql_component": None,
        "failed_statement_index": None,
        "failed_statement_sql": "",
        "execution_error_info": {},
        "classification": classification,
        "repair_strategy": strategy,
        "repair_attempted": True,
        "repair_validation_result": {
            "result": "passed",
            "reason_code": "repair_evidence_valid",
            "details": {"evidence": evidence},
        },
    }

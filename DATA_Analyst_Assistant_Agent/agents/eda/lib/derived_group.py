from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

from DATA_Analyst_Assistant_Agent.shared.expression_gate import validate_expression

DEFAULT_MIN_ENTITY_COUNT = 10


def compute_derived_group_comparison(df: pd.DataFrame, user_question: str, llm: Any = None) -> dict[str, Any] | None:
    prepared = build_derived_group_frame(df, user_question, llm)
    return None if prepared is None else prepared["metadata"]


def build_derived_group_frame(df: pd.DataFrame, user_question: str, llm: Any = None) -> dict[str, Any] | None:
    """Build a temporary entity-level summary for branch questions.

    엔티티(판매자 등)/관측/지표/타겟 컬럼은 기존처럼 컬럼명·질문 문구 기반 휴리스틱으로
    고른다(그대로 유지 — 이번 일반화 대상이 아님). 다만 "어떤 조건으로 부분집합만
    남길지"는 더 이상 정규식으로 미리 정해둔 패턴("N건 이상" 등)만 잡지 않는다(2026-07-23) —
    LLM에게 이 지시사항이 필터 요청인지부터 판단시키고, 맞으면 pandas 불리언 표현식을
    직접 쓰게 한 뒤 shared/expression_gate.py로 검증해서 실행한다. 표현이 뭐든("30건
    이상", "평균 이상", "상위 10% 제외" 등) 같은 경로 하나로 처리되고, 필터 요청이
    아니면(차트 요청 등) None을 돌려줘 이 메커니즘 자체가 개입하지 않는다.
    """
    question = (user_question or "").casefold()
    if "[추가 지시사항]" not in user_question:
        return None
    if llm is None:
        return None

    entity_col = _pick_entity_col(df, question)
    observation_col = _pick_observation_col(df, entity_col)
    metric_col = _pick_metric_col(df, question)
    target_col = _pick_target_col(df, question, metric_col)
    if not entity_col or not metric_col or not target_col or metric_col == target_col:
        return None

    work = df[[c for c in [entity_col, observation_col, metric_col, target_col] if c]].copy()
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[entity_col, metric_col, target_col])
    if work.empty:
        return None

    count_name = "observation_count"
    metric_avg = f"avg_{metric_col}"
    target_avg = f"avg_{target_col}"
    aggregations: dict[str, tuple[str, str]] = {
        count_name: (observation_col or entity_col, "nunique" if observation_col else "size"),
        metric_avg: (metric_col, "mean"),
        target_avg: (target_col, "mean"),
    }
    entity_summary = work.groupby(entity_col, dropna=True).agg(**aggregations).reset_index()

    filter_expression = _llm_filter_expression(
        llm, user_question, columns=[count_name, metric_avg, target_avg])
    if filter_expression is None:
        # LLM이 "필터 요청 아님"으로 판단(차트 요청 등) — 이 메커니즘을 아예 적용하지 않는다.
        return None

    eligible = _apply_filter_expression(entity_summary, filter_expression)
    if eligible is None:
        # 표현식이 게이트를 통과했어도 실행 중 오류가 나면(예: 컬럼 오타) 안전하게
        # 필터 없이 전체를 쓴다 — 지어낸 결과를 내느니 전체를 보여주는 쪽을 택한다.
        eligible = entity_summary.copy()
        filter_expression = ""

    result: dict[str, Any] = {
        "status": "success",
        "kind": "entity_group_comparison",
        "entity_col": entity_col,
        "observation_col": observation_col,
        "metric_col": metric_col,
        "target_col": target_col,
        "count_col": count_name,
        "filter_expression": filter_expression,
        "total_entities": int(len(entity_summary)),
        "eligible_entities": int(len(eligible)),
        "excluded_entities": int(len(entity_summary) - len(eligible)),
        "relationships": {
            "all_entities": _relationship(entity_summary, metric_avg, target_avg),
            "eligible_entities": _relationship(eligible, metric_avg, target_avg),
        },
        "top_entities_by_metric": _records(
            entity_summary.sort_values(metric_avg, ascending=False).head(10),
            [entity_col, count_name, metric_avg, target_avg],
        ),
        "findings": [],
    }
    result["findings"] = _findings(result)

    return {
        "dataframe": eligible.reset_index(drop=True),
        "metadata": result,
        "mart_design": {
            "key_columns": [entity_col],
            "dimension_columns": [entity_col],
            "measure_columns": [count_name, metric_avg, target_avg],
        },
        "key_col": entity_col,
        "measure_cols": [count_name, metric_avg, target_avg],
        "target_col": target_avg,
    }


def _llm_filter_expression(llm: Any, user_question: str, columns: list[str]) -> str | None:
    """이 분기 지시사항이 엔티티를 조건으로 걸러내라는 요청인지 판단하고, 맞으면

    pandas 불리언 표현식 하나로 받는다. 필터 요청이 아니거나(차트 요청 등), LLM 호출이
    실패하거나, 표현식이 안전검사를 통과 못 하면 None을 돌려준다(호출부가 이 메커니즘을
    아예 적용하지 않도록).
    """
    prompt = f"""아래는 데이터 분석 파이프라인의 분기(재분석) 지시사항이다. 이 지시사항이
"개체(예: 판매자)를 어떤 조건으로 걸러내서 그 부분집합만 다시 보고 싶다"는 요청인지 판단하라.
차트 종류 변경, 다른 지표로의 전환, 그 외 특정 조건으로 부분집합을 만드는 것과 무관한
요청이면 필터 요청이 아니다.

[지시사항] {user_question}

필터 요청이면, 이미 개체 하나당 한 행으로 집계된 표에 아래 컬럼만 있다고 가정하고
pandas 불리언 표현식 하나를 써라(새 컬럼을 만들지 말고 이 컬럼만 사용하라):
{columns}

표현식 예시: df["{columns[0]}"] >= 30 / df["{columns[1]}"] >= df["{columns[1]}"].mean() /
df["{columns[1]}"] < df["{columns[1]}"].quantile(0.9)

반드시 JSON만 출력하라(코드블록 없이):
{{"is_filter_request": true 또는 false, "filter_expression": "표현식 또는 null"}}"""
    try:
        raw = llm.invoke(prompt).content
    except Exception:  # noqa: BLE001 — LLM 호출 실패는 "필터 요청 아님"과 동일하게 처리
        return None
    parsed = _parse_json(raw)
    if not parsed or not parsed.get("is_filter_request"):
        return None
    expression = str(parsed.get("filter_expression") or "").strip()
    if not expression:
        return None
    gate = validate_expression(expression, columns)
    if not gate.ok:
        return None
    return expression


def _apply_filter_expression(entity_summary: pd.DataFrame, expression: str) -> pd.DataFrame | None:
    try:
        mask = eval(expression, {"__builtins__": {}}, {"df": entity_summary, "pd": pd, "np": np})  # noqa: S307 — 실행 전 validate_expression 게이트 통과 필수
        return entity_summary[mask].copy()
    except Exception:  # noqa: BLE001 — 실행 중 오류는 호출부가 무필터 폴백으로 처리
        return None


def _parse_json(raw: str) -> dict[str, Any] | None:
    text = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.MULTILINE).strip()
    start = text.find("{")
    if start < 0:
        return None
    for end in range(len(text), start, -1):
        try:
            parsed = json.loads(text[start:end])
            break
        except json.JSONDecodeError:
            continue
    else:
        return None
    return parsed if isinstance(parsed, dict) else None


def _pick_entity_col(df: pd.DataFrame, question: str) -> str | None:
    columns = list(df.columns)
    lowered = {c.casefold(): c for c in columns}
    if ("seller" in question or "판매자" in question) and "seller_id" in lowered:
        return lowered["seller_id"]
    mentioned = [original for lowered_name, original in lowered.items() if lowered_name in question]
    id_mentions = [c for c in mentioned if c.casefold().endswith("_id")]
    for col in id_mentions:
        if col not in {"order_id", "review_id"}:
            return col
    id_cols = [c for c in columns if c.casefold().endswith("_id") and c not in {"order_id", "review_id"}]
    return id_cols[0] if id_cols else None


def _pick_observation_col(df: pd.DataFrame, entity_col: str | None) -> str | None:
    if "order_id" in df.columns:
        return "order_id"
    id_cols = [c for c in df.columns if c.casefold().endswith("_id") and c != entity_col]
    return id_cols[0] if id_cols else None


def _pick_metric_col(df: pd.DataFrame, question: str) -> str | None:
    lowered = {c.casefold(): c for c in df.columns}
    if ("배송" in question or "delivery" in question) and "delivery_days" in lowered:
        return lowered["delivery_days"]
    numeric = _numeric_columns(df)
    for col in numeric:
        if col.casefold() in question:
            return col
    for col in numeric:
        name = col.casefold()
        if not any(token in name for token in ("score", "review", "rating", "count", "flag", "is_")):
            return col
    return numeric[0] if numeric else None


def _pick_target_col(df: pd.DataFrame, question: str, metric_col: str | None) -> str | None:
    lowered = {c.casefold(): c for c in df.columns}
    if any(token in question for token in ("리뷰", "평점", "review", "rating", "score")):
        if "review_score" in lowered:
            return lowered["review_score"]
    numeric = [c for c in _numeric_columns(df) if c != metric_col]
    for col in numeric:
        if col.casefold() in question:
            return col
    for col in numeric:
        if any(token in col.casefold() for token in ("score", "review", "rating")):
            return col
    return numeric[0] if numeric else None


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include="number").columns)


def _relationship(df: pd.DataFrame, metric_col: str, target_col: str) -> dict[str, Any]:
    if df.empty or len(df) < 2:
        return {"n": int(len(df)), "pearson": None, "spearman": None}
    return {
        "n": int(len(df)),
        "pearson": _rounded_corr(df[metric_col], df[target_col], "pearson"),
        "spearman": _rounded_corr(df[metric_col], df[target_col], "spearman"),
        "mean_metric": _rounded(df[metric_col].mean()),
        "mean_target": _rounded(df[target_col].mean()),
    }


def _rounded_corr(left: pd.Series, right: pd.Series, method: str) -> float | None:
    value = left.corr(right, method=method)
    return None if pd.isna(value) else round(float(value), 4)


def _records(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    records = df[columns].where(pd.notna(df[columns]), None).to_dict(orient="records")
    return [{key: _json_value(value) for key, value in row.items()} for row in records]


def _json_value(value: Any) -> Any:
    if isinstance(value, float):
        return _rounded(value)
    return value


def _rounded(value: float) -> float | None:
    return None if pd.isna(value) else round(float(value), 4)


def _findings(result: dict[str, Any]) -> list[str]:
    entity = result["entity_col"]
    metric = result["metric_col"]
    target = result["target_col"]
    expr = result.get("filter_expression") or ""
    filter_desc = f'"{expr}" 조건' if expr else "필터 없이 전체"
    findings = [
        f"{entity}별 집계 표에 {filter_desc}을 적용했습니다.",
        f"전체 {result['total_entities']}개 entity 중 {result['eligible_entities']}개가 필터를 통과했습니다.",
    ]
    eligible_rel = result["relationships"]["eligible_entities"]
    if eligible_rel["spearman"] is not None:
        findings.append(
            f"필터 후 {metric} 평균과 {target} 평균의 Spearman 상관은 {eligible_rel['spearman']}입니다."
        )
    return findings

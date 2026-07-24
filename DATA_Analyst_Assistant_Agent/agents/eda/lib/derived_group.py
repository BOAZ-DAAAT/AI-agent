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
    """Build a temporary frame for branch questions — entity-level summary, or a row filter.

    분기 지시사항은 먼저 (a) 개체(판매자 등) 단위 비교 (b) 개체 집계 없는 단순 행 필터
    (c) 무관, 세 가지로 분류된다(2026-07-24 — 그전까지는 무조건 (a) 경로로만 처리되어
    "이상치인 행만 제외" 같은 (b) 요청이 개체 집계로 잘못 우회되거나, 이 마트처럼 개체
    ID 컬럼이 아예 없으면 조용히 무시되는 문제가 있었다). (b)면 개체 집계를 거치지 않고
    원본 df 그레인 그대로 조건에 맞는 행만 남긴다. (a)일 때만 엔티티/관측/지표/타겟
    컬럼을 컬럼명·질문 문구 기반 휴리스틱으로 고르고, 필터 표현식은 집계된 요약 컬럼
    기준으로 LLM이 다시 판단한다. 표현이 뭐든 shared/expression_gate.py로 검증 후
    실행하고, (c)면 이 메커니즘 자체가 개입하지 않는다.
    """
    question = (user_question or "").casefold()
    if "[추가 지시사항]" not in user_question:
        return None
    if llm is None:
        return None

    classification = _classify_branch_request(llm, user_question, list(df.columns))
    if classification["request_type"] == "unrelated":
        return None
    if classification["request_type"] == "row_filter":
        return _build_row_filter_frame(df, classification.get("filter_expression"))

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


def _classify_branch_request(llm: Any, user_question: str, raw_columns: list[str]) -> dict[str, Any]:
    """분기 지시사항을 (a) 개체 단위 비교 (b) 개체 집계 없는 단순 행 필터 (c) 무관으로 분류한다.

    (b)로 판단되면 원본 df 컬럼만 사용하는 pandas 불리언 표현식까지 함께 받는다 — 이후
    개체 집계를 거치지 않고 그 표현식을 원본 df에 바로 적용한다.
    """
    prompt = f"""아래는 데이터 분석 파이프라인의 분기(재분석) 지시사항이다. 이 지시사항의 성격을 판단하라.

[지시사항] {user_question}

세 가지 중 하나로 분류하라:
- "entity_comparison": 개체(예: 판매자, 고객)를 단위로 그 개체들을 조건으로 걸러내고 개체
  단위 집계로 비교하고 싶다는 요청(예: "주문 10건 이상인 판매자만", "구매액 상위 고객군").
- "row_filter": 개체 집계 없이 지금 있는 행(주문 등) 중 조건을 만족하는 행만 남기거나
  제외하고 싶다는 요청(예: "이상치를 제거", "상위 1%를 제외", "특정 구간만") — 집계 단위를
  바꾸지 않는다.
- "unrelated": 위 둘 다 아님(차트 종류 변경, 다른 지표로 전환 등).

row_filter로 판단되면, 아래 원본 컬럼만 사용해서 pandas 불리언 표현식 하나를 함께 써라
(새 컬럼을 만들지 말고 이 컬럼들만 사용하라):
{raw_columns}

이상치 제외처럼 구체적 기준이 지시사항에 없으면 1.5 IQR을 보편적 기준으로 사용하라.
표현식 예시: df["col"].between(df["col"].quantile(0.25) - 1.5 * (df["col"].quantile(0.75) -
df["col"].quantile(0.25)), df["col"].quantile(0.75) + 1.5 * (df["col"].quantile(0.75) -
df["col"].quantile(0.25)))

반드시 JSON만 출력하라(코드블록 없이):
{{"request_type": "entity_comparison 또는 row_filter 또는 unrelated 중 하나",
  "filter_expression": "row_filter일 때만 표현식 문자열, 아니면 null"}}"""
    try:
        raw = llm.invoke(prompt).content
    except Exception:  # noqa: BLE001 — LLM 호출 실패는 "무관"과 동일하게 처리
        return {"request_type": "unrelated", "filter_expression": None}
    parsed = _parse_json(raw) or {}
    request_type = parsed.get("request_type")
    if request_type not in {"entity_comparison", "row_filter", "unrelated"}:
        return {"request_type": "unrelated", "filter_expression": None}
    return {"request_type": request_type, "filter_expression": parsed.get("filter_expression")}


def _build_row_filter_frame(df: pd.DataFrame, expression: str | None) -> dict[str, Any] | None:
    """개체 집계 없이, 원본 행 그레인을 유지한 채 조건에 맞는 행만 남긴다."""
    if not expression:
        return None
    gate = validate_expression(expression, list(df.columns))
    if not gate.ok:
        return None
    eligible = _apply_filter_expression(df, expression)
    if eligible is None or eligible.empty:
        # 표현식이 게이트를 통과했어도 실행 오류·전량 제외면 지어낸 결과를 내느니 개입 안 한다.
        return None

    result: dict[str, Any] = {
        "status": "success",
        "kind": "row_filter",
        "filter_expression": expression,
        "total_rows": int(len(df)),
        "eligible_rows": int(len(eligible)),
        "excluded_rows": int(len(df) - len(eligible)),
        "findings": [
            f'"{expression}" 조건으로 원본 행을 필터링했습니다.',
            f"전체 {len(df)}행 중 {len(eligible)}행이 조건을 통과했습니다.",
        ],
    }
    return {
        "dataframe": eligible.reset_index(drop=True),
        "metadata": result,
        "mart_design": {},
        "key_col": None,
        "measure_cols": None,
        "target_col": None,
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

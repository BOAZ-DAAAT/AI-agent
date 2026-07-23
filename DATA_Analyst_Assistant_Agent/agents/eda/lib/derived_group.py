from __future__ import annotations

import re
from typing import Any

import pandas as pd


DEFAULT_MIN_ENTITY_COUNT = 10


def compute_derived_group_comparison(df: pd.DataFrame, user_question: str) -> dict[str, Any] | None:
    prepared = build_derived_group_frame(df, user_question)
    return None if prepared is None else prepared["metadata"]


def build_derived_group_frame(df: pd.DataFrame, user_question: str) -> dict[str, Any] | None:
    """Build a temporary entity-level summary for branch questions.

    This is intentionally generic: it detects an entity id, an observation id,
    a metric, and a target from the available dataframe columns and the user's
    wording, then computes groupby summaries without mutating the mart.
    """
    question = (user_question or "").casefold()
    if "[추가 지시사항]" not in user_question:
        return None
    if not _looks_like_entity_group_request(question):
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
    min_count, threshold_source = _requested_min_count(user_question)
    eligible = entity_summary[entity_summary[count_name] >= min_count].copy()

    result: dict[str, Any] = {
        "status": "success",
        "kind": "entity_group_comparison",
        "entity_col": entity_col,
        "observation_col": observation_col,
        "metric_col": metric_col,
        "target_col": target_col,
        "count_col": count_name,
        "min_count": min_count,
        "threshold_source": threshold_source,
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

    group_col = ""
    if _asks_for_high_metric_group(question) and len(eligible) >= 2:
        group_col = f"{metric_col}_group"
        threshold = float(eligible[metric_avg].quantile(0.75))
        eligible[group_col] = f"other_{metric_col}"
        eligible.loc[eligible[metric_avg] >= threshold, group_col] = f"high_{metric_col}"
        result["derived_group_col"] = group_col
        result["high_metric_group"] = _high_metric_group(eligible, metric_avg, target_avg)

    result["findings"] = _findings(result)
    measure_cols = [count_name, metric_avg, target_avg]
    if group_col:
        measure_cols.append(group_col)
    return {
        "dataframe": eligible.reset_index(drop=True),
        "metadata": result,
        "mart_design": {
            "key_columns": [entity_col],
            "dimension_columns": [group_col] if group_col else [entity_col],
            "measure_columns": [count_name, metric_avg, target_avg],
        },
        "key_col": group_col or entity_col,
        "measure_cols": [count_name, metric_avg, target_avg],
        "target_col": target_avg,
    }


def _looks_like_entity_group_request(question: str) -> bool:
    entity_tokens = ("별", "그룹", "판매자", "seller", "entity", "표본", "소표본", "상위", "하위")
    metric_tokens = ("평균", "충분", "제외", "긴", "높은", "낮은", "long", "high", "exclude")
    return any(token in question for token in entity_tokens) and any(token in question for token in metric_tokens)


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


def _requested_min_count(user_question: str) -> tuple[int, str]:
    patterns = [
        r"(?:주문\s*수|표본|n)\s*(\d+)\s*(?:건|개)?\s*(?:이상|>=)",
        r"(\d+)\s*(?:건|개)?\s*(?:이상|>=)\s*(?:판매자|seller|표본|주문)",
        r"(\d+)\s*(?:건|개)?\s*(?:미만|<)\s*(?:판매자|seller|표본|주문)?\s*(?:제외|빼)",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_question)
        if match:
            return max(1, int(match.group(1))), "user_specified"
    return DEFAULT_MIN_ENTITY_COUNT, "default_when_unspecified"


def _asks_for_high_metric_group(question: str) -> bool:
    return any(token in question for token in ("긴", "높은", "상위", "long", "high", "top"))


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


def _high_metric_group(df: pd.DataFrame, metric_col: str, target_col: str) -> dict[str, Any]:
    threshold = float(df[metric_col].quantile(0.75))
    high = df[df[metric_col] >= threshold]
    rest = df[df[metric_col] < threshold]
    return {
        "split": "top_quartile_by_metric",
        "threshold": _rounded(threshold),
        "high_group": _group_stats(high, metric_col, target_col),
        "rest_group": _group_stats(rest, metric_col, target_col),
        "target_mean_gap_high_minus_rest": (
            None
            if high.empty or rest.empty
            else _rounded(high[target_col].mean() - rest[target_col].mean())
        ),
    }


def _group_stats(df: pd.DataFrame, metric_col: str, target_col: str) -> dict[str, Any]:
    return {
        "entities": int(len(df)),
        "mean_metric": None if df.empty else _rounded(df[metric_col].mean()),
        "mean_target": None if df.empty else _rounded(df[target_col].mean()),
    }


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
    min_count = result["min_count"]
    findings = [
        f"{entity}별 {result['count_col']} 기준 {min_count}건 이상 필터를 적용했습니다.",
        f"전체 {result['total_entities']}개 entity 중 {result['eligible_entities']}개가 필터를 통과했습니다.",
    ]
    eligible_rel = result["relationships"]["eligible_entities"]
    if eligible_rel["spearman"] is not None:
        findings.append(
            f"필터 후 {metric} 평균과 {target} 평균의 Spearman 상관은 {eligible_rel['spearman']}입니다."
        )
    high_group = result.get("high_metric_group")
    if high_group:
        gap = high_group.get("target_mean_gap_high_minus_rest")
        findings.append(
            f"{metric} 상위군과 나머지의 {target} 평균 차이(high-rest)는 {gap}입니다."
        )
    return findings

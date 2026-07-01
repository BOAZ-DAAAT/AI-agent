"""insight 노드 — downstream용 statistical_metadata 집계 + LLM 인사이트/구조해석."""

from __future__ import annotations

import json
import math
from typing import Any, Dict

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import append_errors, get_context, get_llm
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.tool_runner import run_node_with_retry
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import insight_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState

try:  # 제공자에 따라 openai 예외가 없을 수 있어 방어적으로 import
    from openai import RateLimitError
except Exception:  # noqa: BLE001
    class RateLimitError(Exception):
        pass


def insight_node(state: EDAState) -> dict:
    ctx = get_context()
    df = ctx.df
    measure_cols = ctx.measure_cols
    key_col = ctx.key_col
    count_col = ctx.count_col

    statistical_metadata: Dict[str, Any] = {}
    data_level: Dict[str, Any] = {}
    cautions: list = []
    if df is not None:
        from DATA_Analyst_Assistant_Agent.agents.eda.lib.missing import detect_missing
        from DATA_Analyst_Assistant_Agent.agents.eda.lib.outlier import detect_outliers_iqr
        from DATA_Analyst_Assistant_Agent.agents.eda.lib.quality import check_duplicates_fn
        from DATA_Analyst_Assistant_Agent.agents.eda.lib.reliability import (
            assess_sample_reliability, build_cautions, detect_data_level,
        )

        numeric_cols = [c for c in (measure_cols or []) if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            numeric_cols = list(df.select_dtypes(include=["float64", "int64"]).columns)

        def _r(x, nd: int = 4):
            """NaN/inf는 None으로(JSON 안전), 나머진 반올림."""
            try:
                x = float(x)
            except (TypeError, ValueError):
                return None
            return round(x, nd) if math.isfinite(x) else None

        dist_stats = {}
        for col in numeric_cols:
            s = df[col].dropna()
            if s.empty:
                dist_stats[col] = {"type": "numeric", "unique_count": 0}
                continue

            q = s.quantile([0.01, 0.05, 0.25, 0.75, 0.95, 0.99])
            iqr = float(q[0.75] - q[0.25])
            lo_fence, hi_fence = q[0.25] - 1.5 * iqr, q[0.75] + 1.5 * iqr
            outlier_rate = float(((s < lo_fence) | (s > hi_fence)).mean()) if iqr > 0 else 0.0
            skew = float(s.skew()) if s.nunique() > 2 else 0.0
            kurt = float(s.kurt()) if s.nunique() > 3 else 0.0
            all_positive = bool(s.min() > 0)       # 엄밀히 양수(log 직접 가능)
            non_negative = bool(s.min() >= 0)
            has_zero = bool((s == 0).any())

            # eda_notes — 수치에서 규칙으로 도출(LLM 없음)
            shape = ("left_skewed" if skew < -0.5 else
                     "right_skewed" if skew > 0.5 else "symmetric")
            suspected = ("low_tail" if skew < -1 else
                         "high_tail" if skew > 1 else
                         "both_tails" if outlier_rate > 0.05 else "none")
            handling: list = []
            if shape == "right_skewed":
                if all_positive:
                    handling.append("log_transform")
                elif non_negative:             # 0 포함 → log(x+1)
                    handling.append("log1p_transform")
            if outlier_rate > 0.03:
                handling.append("avoid_naive_outlier_removal")
            if shape != "symmetric":
                handling.append("prefer_nonparametric_or_transform")

            normality = ("approx_normal" if abs(skew) < 0.5 and abs(kurt) < 1 else
                         "heavy_tailed" if abs(kurt) >= 3 else "skewed")

            dist_stats[col] = {
                "type":             "numeric",
                # semantic_type 은 후속 스텝(휴리스틱)에서 추가 예정
                "mean":             _r(s.mean()),
                "median":           _r(s.median()),
                "std":              _r(s.std()),
                "skewness":         _r(skew),
                "kurtosis":         _r(kurt),
                "min":              _r(s.min()),
                "p01":              _r(q[0.01]),
                "p05":              _r(q[0.05]),
                "q1":               _r(q[0.25]),
                "q3":               _r(q[0.75]),
                "p95":              _r(q[0.95]),
                "p99":              _r(q[0.99]),
                "max":              _r(s.max()),
                "missing_rate":     _r(df[col].isna().mean()),
                "zero_rate":        _r((s == 0).mean()),
                "unique_count":     int(s.nunique()),
                "outlier_rate_iqr": _r(outlier_rate),
                "all_positive":     all_positive,
                "non_negative":     non_negative,
                "has_zero":         has_zero,
                "normality":        normality,
                "eda_notes": {
                    "shape":                shape,
                    "suspected_outliers":   suspected,
                    "recommended_handling": handling,
                },
            }

        # ── 범주형 컬럼 프로파일 (같은 dist_stats에, type으로 구분) ──
        cat_cols = [c for c in df.columns
                    if c not in numeric_cols
                    and not pd.api.types.is_numeric_dtype(df[c])
                    and not pd.api.types.is_datetime64_any_dtype(df[c])]
        for col in cat_cols:
            s = df[col].dropna()
            if s.empty:
                dist_stats[col] = {"type": "categorical", "unique_count": 0}
                continue
            vc = s.value_counts()
            n = len(s)
            nun = int(s.nunique())
            top1_share = float(vc.iloc[0] / n)
            is_id_like = nun > 0.9 * n                        # 거의 다 유니크 = id 같은 것
            rare_count = int((vc < 30).sum())                 # 표본 적은 범주 (n<30)
            rare_share = float(vc[vc < 30].sum() / n)         # 희소범주가 차지하는 데이터 비율
            cardinality = "high" if nun > 50 else "medium" if nun > 10 else "low"
            # balance: 한 범주 지배 / 긴 꼬리(희소 다수) / 균형
            balance = ("dominated" if top1_share > 0.5 else
                       "long_tailed" if nun and rare_count / nun > 0.3 else
                       "balanced")
            handling = []
            if is_id_like:
                handling.append("treat_as_id_or_drop")
            elif cardinality == "high" or balance == "long_tailed":
                handling.append("group_rare_categories")
            if balance == "dominated":
                handling.append("check_dominant_category")

            entry = {
                "type":                    "categorical",
                "unique_count":            nun,
                "missing_rate":            _r(df[col].isna().mean()),
                "is_id_like":              is_id_like,
                "top1_share":              _r(top1_share),
                "top10_coverage":          _r(float(vc.head(10).sum() / n)),
                "rare_category_count":     rare_count,
                "rare_category_threshold": "n < 30",
                "rare_category_share":     _r(rare_share),
                "mode":                    str(vc.index[0]),
                "eda_notes": {
                    "cardinality":          cardinality,
                    "balance":              balance,
                    "recommended_handling": handling,
                },
            }
            if not is_id_like:                                # id는 top값 무의미 → 스킵
                entry["top_values"] = {str(k): int(v) for k, v in vc.head(10).items()}
            dist_stats[col] = entry

        corr_pairs = {}
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    key = f"corr_{numeric_cols[i]}_vs_{numeric_cols[j]}"
                    corr_pairs[key] = round(float(corr.iloc[i, j]), 3)

        missing_info = detect_missing(df)
        outlier_info = detect_outliers_iqr(df, measure_cols=measure_cols)
        dup_info     = check_duplicates_fn(df)
        outliers_by_col = {
            col: v.get("outlier_count", 0)
            for col, v in outlier_info.items()
            if isinstance(v, dict)
        }

        group_comparison = {}
        if key_col and key_col in df.columns:
            for col in numeric_cols:
                try:
                    grp = df.groupby(key_col)[col].mean().dropna()
                    group_comparison[col] = {
                        "top3_groups":    {str(k): round(float(v), 4) for k, v in grp.nlargest(3).items()},
                        "bottom3_groups": {str(k): round(float(v), 4) for k, v in grp.nsmallest(3).items()},
                        "group_max":      round(float(grp.max()), 4),
                        "group_min":      round(float(grp.min()), 4),
                        "group_std":      round(float(grp.std()), 4),
                    }
                except Exception:
                    pass

        # 데이터 한계 자가점검 (코드, LLM 없음): 원본/집계 판정 + 표본 신뢰도 + 주의사항
        data_level = detect_data_level(df, key_col=key_col, numeric_cols=numeric_cols)
        sample_reliability = assess_sample_reliability(
            df, key_col=key_col, count_col=count_col, data_level=data_level.get("level", "unknown"))
        cautions = build_cautions(data_level, sample_reliability, corr_pairs)

        # clustering_result 가 비어있으면(컨트롤러가 안 돌린 경우) skip 처리
        clustering = state.get("clustering_result") or {}
        statistical_metadata = {
            "row_count":          len(df),
            "data_level":         data_level,
            "sample_reliability": sample_reliability,
            "cautions":           cautions,
            "distribution":       dist_stats,
            "group_comparison":   group_comparison,
            "correlation_pairs":  corr_pairs,
            "missing_total":      missing_info.get("total_missing", 0),
            "outliers_by_column": outliers_by_col,
            "duplicate_count":    dup_info.get("duplicate_count", 0),
            "clustering": {
                "n_clusters":        clustering.get("n_clusters"),
                "silhouette_score":  clustering.get("silhouette_score"),
                "cluster_centroids": clustering.get("cluster_centroids", {}),
            } if (clustering and not clustering.get("skip")) else {"skip": True},
        }

    all_results = f"""
[구조 탐색] {state.get('inspect_result', '해당 없음')}
[품질 점검] {state.get('quality_result', '해당 없음')}
[분포 분석] {state.get('distribution_result', '해당 없음')}
[그룹 비교] {state.get('comparison_result', '해당 없음')}
[관계 탐색] {state.get('relationship_result', '해당 없음')}
[시간 분석] {state.get('time_result', '해당 없음')}
[클러스터링] {json.dumps(state.get('clustering_result', {}), ensure_ascii=False)}
"""
    prompt = insight_prompt(state["user_question"], statistical_metadata, all_results)
    fb = state.get("validation_feedback")
    if fb:
        prompt += f"\n[직전 검증 지적 — 반드시 보완하라]\n{fb}\n"

    llm = get_llm()
    try:
        insight_result = llm.invoke(prompt).content.strip()
        err = None
    except Exception as e:  # noqa: BLE001
        if isinstance(e, RateLimitError):
            truncated_results = "\n".join([
                f"[{label}] {text[:300]}..."
                for label, text in [
                    ("구조 탐색", state.get("inspect_result", "")),
                    ("품질 점검", state.get("quality_result", "")),
                    ("분포 분석", state.get("distribution_result", "")),
                    ("그룹 비교", state.get("comparison_result", "")),
                    ("관계 탐색", state.get("relationship_result", "")),
                    ("시간 분석", state.get("time_result", "")),
                ]
                if text
            ])
            slim_prompt = prompt.replace(all_results, truncated_results)
            insight_result, err = run_node_with_retry(
                lambda: llm.invoke(slim_prompt).content.strip(), "insight", fallback="인사이트 생성 실패"
            )
        else:
            insight_result, err = run_node_with_retry(
                lambda: llm.invoke(prompt).content.strip(), "insight", fallback="인사이트 생성 실패"
            )

    # 분석 노드들이 ctx에 누적한 차트 주문서를 state로 노출 + 아티팩트로 영속화.
    chart_requests = list(get_context().chart_requests)
    _persist_chart_requests(chart_requests)

    return {
        "insight_result": insight_result,
        "statistical_metadata": statistical_metadata,
        "data_level": data_level,
        "cautions": cautions,
        "chart_requests": chart_requests,
        "error_log": append_errors(state, err),
    }


def _persist_chart_requests(chart_requests: list) -> None:
    """차트 주문서를 outputs/chart_requests.json 으로 기록(Phase B/report 소비용). 실패해도 무시."""
    import os

    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize  # OUTPUT_DIR 동적 반영

    try:
        out_path = os.path.join(os.path.dirname(visualize.OUTPUT_DIR), "chart_requests.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chart_requests, f, ensure_ascii=False, indent=2)
    except Exception:  # noqa: BLE001
        pass

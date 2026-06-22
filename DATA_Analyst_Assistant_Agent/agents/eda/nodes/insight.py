"""insight 노드 — downstream용 statistical_metadata 집계 + LLM 인사이트/구조해석."""

from __future__ import annotations

import json
from typing import Any, Dict

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import append_errors, get_context, get_llm
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.react import run_node_with_retry
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
        from DATA_Analyst_Assistant_Agent.agents.eda.tools.missing import detect_missing
        from DATA_Analyst_Assistant_Agent.agents.eda.tools.outlier import detect_outliers_iqr
        from DATA_Analyst_Assistant_Agent.agents.eda.tools.quality import check_duplicates_fn
        from DATA_Analyst_Assistant_Agent.agents.eda.tools.reliability import (
            assess_sample_reliability, build_cautions, detect_data_level,
        )

        numeric_cols = [c for c in (measure_cols or []) if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            numeric_cols = list(df.select_dtypes(include=["float64", "int64"]).columns)

        dist_stats = {}
        for col in numeric_cols:
            s = df[col].dropna()
            dist_stats[col] = {
                "mean":     round(float(s.mean()), 4),
                "median":   round(float(s.median()), 4),
                "std":      round(float(s.std()), 4),
                "skewness": round(float(s.skew()), 4),
                "min":      round(float(s.min()), 4),
                "max":      round(float(s.max()), 4),
            }

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

    from DATA_Analyst_Assistant_Agent.agents.eda.tools import visualize  # OUTPUT_DIR 동적 반영

    try:
        out_path = os.path.join(os.path.dirname(visualize.OUTPUT_DIR), "chart_requests.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chart_requests, f, ensure_ascii=False, indent=2)
    except Exception:  # noqa: BLE001
        pass

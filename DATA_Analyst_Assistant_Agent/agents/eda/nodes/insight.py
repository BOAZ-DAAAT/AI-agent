"""insight 노드 — downstream용 statistical_metadata 집계 + LLM 인사이트/구조해석."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import (
    append_errors, get_context, get_llm, split_marked_json,
)
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.tool_runner import run_node_with_retry
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import insight_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.prompts.insight import SUMMARY_FACTS_MARKER
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState
from DATA_Analyst_Assistant_Agent.agents.sql.validator.integrity_loader import load_scoped_integrity_text


_GRAIN_TOKEN = re.compile(r"[a-z_]{3,}")
_GRAIN_STOPWORDS = {"one", "row", "rows", "per", "the", "and", "grain", "level", "table", "unit"}


def _grain_conflicts(declared_grain: str, detected_grain_hint: str) -> bool:
    """SQL 선언 grain과 EDA 자체 추정 grain이 서로 다른 키를 가리키는지 보수적으로 판단.

    둘에서 식별자 토큰(영문 소문자 3자+)을 뽑아, 선언 grain에 토큰이 있는데 추정 grain과 하나도
    겹치지 않으면 충돌로 본다. 선언 grain이 비었거나 식별자 토큰이 없으면(한글 서술 등) 충돌 아님(폴백).
    """
    if not declared_grain:
        return False
    declared = {t for t in _GRAIN_TOKEN.findall(declared_grain.lower()) if t not in _GRAIN_STOPWORDS}
    detected = {t for t in _GRAIN_TOKEN.findall((detected_grain_hint or "").lower()) if t not in _GRAIN_STOPWORDS}
    # 한쪽이라도 식별자 토큰이 없으면 '불일치'가 아니라 '비교 불가' → false positive 방지(#130).
    if not declared or not detected:
        return False
    return declared.isdisjoint(detected)


def _fail_tables_from_integrity_text(text: str) -> list[str]:
    """스코핑된 정합성 텍스트에서 이슈가 난 테이블명만 추출(중복 제거·순서 보존)."""
    tables: list[str] = []
    for match in re.finditer(r"\[[A-Z_]+\]\s+([A-Za-z_][A-Za-z0-9_]*)", text):
        name = match.group(1)
        if name != "dataset" and name not in tables:
            tables.append(name)
    return tables

try:  # 제공자에 따라 openai 예외가 없을 수 있어 방어적으로 import
    from openai import RateLimitError
except Exception:  # noqa: BLE001
    class RateLimitError(Exception):
        pass


def _jround(x, nd: int = 4):
    """NaN/inf는 None으로(JSON 안전), 나머진 반올림. (insight 계산 함수 공용)"""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def compute_group_comparison(df, key_col, numeric_cols) -> Dict[str, Any]:
    """key_col 그룹별 수치 비교 + 효과크기. 순수 계산(LLM 없음).

    각 수치 컬럼: top3/bottom3·group_std·n_groups·min/max_group_n +
      eta_squared(그룹당 복수 관측=raw일 때만, SS_between/SS_total; 집계본이면 skipped_aggregated)
      spread_ratio(group_min>0일 때)·cv_across_groups(group_mean≠0일 때).
    """
    group_comparison: Dict[str, Any] = {}
    if not (key_col and key_col in df.columns):
        return group_comparison
    for col in numeric_cols:
        try:
            g = df.groupby(key_col)[col]
            grp = g.mean().dropna()                        # 그룹별 평균
            if grp.empty:
                continue
            counts = g.count().reindex(grp.index)          # 그룹별 표본 수(col 기준 non-NaN)
            group_mean = float(grp.mean())
            group_max = float(grp.max())
            group_min = float(grp.min())
            group_std = float(grp.std())                   # 그룹 평균들의 표준편차(그룹 1개면 NaN)
            max_n = int(counts.max())

            entry = {
                "top3_groups":    {str(k): round(float(v), 4) for k, v in grp.nlargest(3).items()},
                "bottom3_groups": {str(k): round(float(v), 4) for k, v in grp.nsmallest(3).items()},
                "group_max":      round(group_max, 4),
                "group_min":      round(group_min, 4),
                "group_std":      _jround(group_std, 4),
                "n_groups":       int(len(grp)),
                "min_group_n":    int(counts.min()),
                "max_group_n":    max_n,
            }

            # 효과크기 eta² — 그룹당 복수 관측(raw)일 때만 의미. 집계본(그룹당 1행)이면 스킵.
            if max_n >= 2:
                grand = float(df[col].mean())
                ss_total = float(((df[col] - grand) ** 2).sum())
                ss_between = float((counts * (grp - grand) ** 2).sum())
                eta = ss_between / ss_total if ss_total > 0 else None
                entry["eta_squared"] = _jround(eta, 4)
                if eta is None:
                    entry["eta_interpretation"] = "undefined"
                elif eta < 0.06:
                    entry["eta_interpretation"] = "small"
                elif eta < 0.14:
                    entry["eta_interpretation"] = "medium"
                else:
                    entry["eta_interpretation"] = "large"
            else:
                entry["eta_squared"] = None
                entry["eta_interpretation"] = "skipped_aggregated"

            entry["spread_ratio"] = _jround(group_max / group_min, 4) if group_min > 0 else None
            entry["cv_across_groups"] = _jround(group_std / group_mean, 4) if abs(group_mean) > 1e-9 else None
            group_comparison[col] = entry
        except Exception:  # noqa: BLE001
            pass
    return group_comparison


# ─────────────────────────────
# semantic_type 분류 (순수 코드, 토큰 0)
# dtype(numeric/categorical) 위에 컬럼의 '의미 역할'을 이름·값 신호로 추론한다.
# 이름+값 일치=high, 이름·값 불일치나 한쪽만=medium, 값도 이름도 약함=low.
# 이름 추론은 휴리스틱이라 confidence 필수 — 분석이 신뢰도 보고 소비한다.
# ─────────────────────────────
_ID_NAME = re.compile(r"(^|_)(id|uuid|guid)($|_)", re.IGNORECASE)
_MONETARY_KEYS = ("price", "value", "payment", "amount", "cost", "revenue",
                  "freight", "sales", "gmv", "profit", "fee", "_brl", "monetary")
_RATING_KEYS = ("score", "rating", "review", "star", "csat", "nps")
_COUNT_KEYS = ("count", "qty", "quantity", "num_", "n_", "_cnt", "freq", "items",
               "orders", "units", "transactions", "sessions")
_DURATION_KEYS = ("days", "duration", "elapsed", "latency", "tenure",
                  "age", "delay", "_time", "hours", "minutes", "seconds")


def _name_signal_numeric(name: str) -> str | None:
    """컬럼 이름으로 의미 역할 추정(id 우선). 없으면 None."""
    low = name.casefold()
    if _ID_NAME.search(low):
        return "id"
    if any(k in low for k in _MONETARY_KEYS):
        return "monetary"
    if any(k in low for k in _RATING_KEYS):
        return "rating"
    if any(k in low for k in _COUNT_KEYS):
        return "count"
    if any(k in low for k in _DURATION_KEYS):
        return "duration"
    return None


def classify_semantic_numeric(col, s, unique_count, all_positive, non_negative):
    """수치형 컬럼의 의미 역할 + confidence. 반환 (semantic_type, confidence)."""
    name_type = _name_signal_numeric(col)
    try:
        is_int = bool((s % 1 == 0).all())
    except TypeError:
        is_int = False
    n = len(s)
    value_type = None
    if is_int and n and unique_count > 0.9 * n:
        value_type = "id"
    elif is_int and float(s.min()) >= 1 and float(s.max()) <= 10 and unique_count <= 10:
        value_type = "rating"
    elif is_int and non_negative:
        value_type = "count"
    elif all_positive:
        value_type = "monetary"

    if name_type and value_type:
        return (name_type, "high") if name_type == value_type else (name_type, "medium")
    if name_type:
        return name_type, "medium"
    if value_type:
        return value_type, "medium"
    return "generic", "low"


def classify_semantic_categorical(col, is_id_like):
    """범주형 컬럼의 의미 역할 + confidence. 집계본 group_key는 조립부에서 별도 교정."""
    name_id = bool(_ID_NAME.search(col.casefold()))
    if is_id_like:
        return "id", ("high" if name_id else "medium")
    if name_id:
        return "id", "low"   # 이름은 id인데 값이 유니크하지 않음 → 약한 신호
    return "category", "high"


def compute_numeric_distribution(df, numeric_cols) -> Dict[str, Any]:
    """수치형 컬럼 분포 통계. 순수 계산(LLM 없음).
    percentile·skewness·kurtosis·outlier_rate·all_positive·normality·semantic_type + eda_notes.
    """
    dist_stats: Dict[str, Any] = {}
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
        all_positive = bool(s.min() > 0)
        non_negative = bool(s.min() >= 0)
        has_zero = bool((s == 0).any())

        shape = ("left_skewed" if skew < -0.5 else
                 "right_skewed" if skew > 0.5 else "symmetric")
        suspected = ("low_tail" if skew < -1 else
                     "high_tail" if skew > 1 else
                     "both_tails" if outlier_rate > 0.05 else "none")
        handling: list = []
        if shape == "right_skewed":
            if all_positive:
                handling.append("log_transform")
            elif non_negative:
                handling.append("log1p_transform")
        if outlier_rate > 0.03:
            handling.append("avoid_naive_outlier_removal")
        if shape != "symmetric":
            handling.append("prefer_nonparametric_or_transform")

        normality = ("approx_normal" if abs(skew) < 0.5 and abs(kurt) < 1 else
                     "heavy_tailed" if abs(kurt) >= 3 else "skewed")

        semantic_type, semantic_confidence = classify_semantic_numeric(
            col, s, int(s.nunique()), all_positive, non_negative)

        # semantic_type 기반 handling 교정: 분포 모양만으론 틀리는 추천을 의미로 바로잡는다.
        # rating=순서형(log·이상치제거 부적합), id=분석 대상 아님.
        if semantic_type == "rating":
            handling = ["treat_as_ordinal", "avoid_outlier_removal", "use_rank_or_nonparametric"]
        elif semantic_type == "id":
            handling = ["exclude_from_analysis"]

        dist_stats[col] = {
            "type":             "numeric",
            "semantic_type":       semantic_type,
            "semantic_confidence": semantic_confidence,
            "mean":             _jround(s.mean()),
            "median":           _jround(s.median()),
            "std":              _jround(s.std()),
            "skewness":         _jround(skew),
            "kurtosis":         _jround(kurt),
            "min":              _jround(s.min()),
            "p01":              _jround(q[0.01]),
            "p05":              _jround(q[0.05]),
            "q1":               _jround(q[0.25]),
            "q3":               _jround(q[0.75]),
            "p95":              _jround(q[0.95]),
            "p99":              _jround(q[0.99]),
            "max":              _jround(s.max()),
            "missing_rate":     _jround(df[col].isna().mean()),
            "zero_rate":        _jround((s == 0).mean()),
            "unique_count":     int(s.nunique()),
            "outlier_rate_iqr": _jround(outlier_rate),
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
    return dist_stats


def compute_categorical_distribution(df, numeric_cols) -> Dict[str, Any]:
    """범주형 컬럼 프로파일. 순수 계산(LLM 없음).
    cardinality·top1_share·rare범주·is_id_like·balance + eda_notes(handling). id는 top_values 스킵.
    """
    out: Dict[str, Any] = {}
    cat_cols = [c for c in df.columns
                if c not in numeric_cols
                and not pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_datetime64_any_dtype(df[c])]
    for col in cat_cols:
        s = df[col].dropna()
        if s.empty:
            out[col] = {"type": "categorical", "unique_count": 0}
            continue
        vc = s.value_counts()
        n = len(s)
        nun = int(s.nunique())
        top1_share = float(vc.iloc[0] / n)
        is_id_like = nun > 0.9 * n
        rare_count = int((vc < 30).sum())
        rare_share = float(vc[vc < 30].sum() / n)
        cardinality = "high" if nun > 50 else "medium" if nun > 10 else "low"
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

        semantic_type, semantic_confidence = classify_semantic_categorical(col, is_id_like)

        entry = {
            "type":                    "categorical",
            "semantic_type":           semantic_type,
            "semantic_confidence":     semantic_confidence,
            "unique_count":            nun,
            "missing_rate":            _jround(df[col].isna().mean()),
            "is_id_like":              is_id_like,
            "top1_share":              _jround(top1_share),
            "top10_coverage":          _jround(float(vc.head(10).sum() / n)),
            "rare_category_count":     rare_count,
            "rare_category_threshold": "n < 30",
            "rare_category_share":     _jround(rare_share),
            "mode":                    str(vc.index[0]),
            "eda_notes": {
                "cardinality":          cardinality,
                "balance":              balance,
                "recommended_handling": handling,
            },
        }
        if not is_id_like:
            entry["top_values"] = {str(k): int(v) for k, v in vc.head(10).items()}
        out[col] = entry
    return out


def compute_correlation_pairs(df, numeric_cols) -> Dict[str, Any]:
    """수치형 컬럼 쌍의 관계 객체. 순수 계산(LLM 없음).
    pearson·spearman·nonlinearity + 강한 쌍(|r|>=0.2 & n>=30)엔 r_squared·binned_trend(산점도 압축).
    """
    corr_pairs: Dict[str, Any] = {}
    if len(numeric_cols) < 2:
        return corr_pairs
    pear = df[numeric_cols].corr()
    spear = df[numeric_cols].corr(method="spearman")
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            a, b = numeric_cols[i], numeric_cols[j]
            p = float(pear.iloc[i, j])
            sp = float(spear.iloc[i, j])
            gap = abs(sp) - abs(p)
            nonlin = ("weak" if abs(p) < 0.1 and abs(sp) < 0.1 else
                      "monotonic_nonlinear" if gap > 0.15 else "linear")
            pair_df = df[[a, b]].dropna()
            npts = len(pair_df)
            entry = {
                "pearson_r":    _jround(p, 3),
                "spearman_r":   _jround(sp, 3),
                "nonlinearity": nonlin,
                "n":            npts,
            }
            if npts >= 30 and (abs(p) >= 0.2 or abs(sp) >= 0.2):
                entry["r_squared_linear"] = _jround(p * p, 3)
                try:
                    x = pair_df[a].astype(float)
                    y = pair_df[b].astype(float)
                    bins = pd.qcut(x, q=min(10, max(2, x.nunique())), duplicates="drop")
                    bt = []
                    for interval, grp in y.groupby(bins, observed=True):
                        bt.append({
                            "x_range":  [_jround(interval.left, 2), _jround(interval.right, 2)],
                            "y_median": _jround(grp.median(), 3),
                            "y_iqr":    [_jround(grp.quantile(0.25), 3), _jround(grp.quantile(0.75), 3)],
                            "n":        int(len(grp)),
                        })
                    entry["binned_trend"] = bt
                    entry["binning"] = {"method": "quantile", "n_bins": len(bt)}
                except Exception:  # noqa: BLE001
                    pass
            corr_pairs[f"corr_{a}_vs_{b}"] = entry
    return corr_pairs


def insight_node(state: EDAState) -> dict:
    ctx = get_context()
    df = ctx.df
    measure_cols = ctx.measure_cols
    key_col = ctx.key_col
    count_col = ctx.count_col

    statistical_metadata: Dict[str, Any] = {}
    data_level: Dict[str, Any] = {}
    cautions: list = []
    analysis_constraints: list = []
    if df is not None:
        from DATA_Analyst_Assistant_Agent.agents.eda.lib.missing import detect_missing
        from DATA_Analyst_Assistant_Agent.agents.eda.lib.outlier import detect_outliers_iqr
        from DATA_Analyst_Assistant_Agent.agents.eda.lib.quality import check_duplicates_fn
        from DATA_Analyst_Assistant_Agent.agents.eda.lib.reliability import (
            assess_sample_reliability, build_analysis_constraints, build_cautions, detect_data_level,
        )

        numeric_cols = [c for c in (measure_cols or []) if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            numeric_cols = list(df.select_dtypes(include=["float64", "int64"]).columns)

        dist_stats = compute_numeric_distribution(df, numeric_cols)

        dist_stats.update(compute_categorical_distribution(df, numeric_cols))

        corr_pairs = compute_correlation_pairs(df, numeric_cols)

        missing_info = detect_missing(df)
        outlier_info = detect_outliers_iqr(df, measure_cols=measure_cols)
        dup_info     = check_duplicates_fn(df)
        outliers_by_col = {
            col: v.get("outlier_count", 0)
            for col, v in outlier_info.items()
            if isinstance(v, dict)
        }

        group_comparison = compute_group_comparison(df, key_col, numeric_cols)

        # 데이터 한계 자가점검 (코드, LLM 없음): 원본/집계 판정 + 표본 신뢰도 + 주의사항
        data_level = detect_data_level(df, key_col=key_col, numeric_cols=numeric_cols)
        sample_reliability = assess_sample_reliability(
            df, key_col=key_col, count_col=count_col, data_level=data_level.get("level", "unknown"))

        # SQL 선언 grain vs EDA 자체 추정 grain 교차검증(#130). EDA는 통계 휴리스틱만으로 grain을
        # 추정해 오판할 수 있는데(near-unique를 1:1로 착각), SQL이 선언한 grain과 다르면 신뢰도를
        # 낮추고 구조체 caution 을 남긴다. plan_business_grain 이 비면 조용히 스킵(폴백).
        declared_grain = str(state.get("plan_business_grain") or "").strip()
        grain_mismatch = _grain_conflicts(declared_grain, data_level.get("grain_hint", ""))
        if grain_mismatch:
            data_level["confidence"] = "low"
            data_level["grain_conflict_with_sql"] = declared_grain

        # 집계본의 group key는 ID가 아니라 '묶는 기준' — classifier가 near-unique라 id로 오판하는 걸
        # 구조 사실(집계본 key_col)로 교정한다. semantic_type 체계의 정식 일부(집계 여부는 구조라 high).
        if data_level.get("is_aggregated") and key_col and isinstance(dist_stats.get(key_col), dict):
            gk = dist_stats[key_col]
            gk["semantic_type"] = "group_key"
            gk["semantic_confidence"] = "high"
            gk["is_id_like"] = False
            if isinstance(gk.get("eda_notes"), dict):
                gk["eda_notes"]["recommended_handling"] = ["use_as_group_key"]

        # rule 기반(결정론): 구조체 cautions + hard 계약(analysis_constraints)
        cautions = build_cautions(data_level, sample_reliability, corr_pairs, distribution=dist_stats)
        analysis_constraints = build_analysis_constraints(data_level)

        # (#130) 상류 SQL 정합성/grain 컨텍스트를 구조체 caution 으로 남긴다 — 프롬프트뿐 아니라
        # eda_summary.json 의 cautions 배열에 실려 하류(분석/리포트)가 구조적으로 읽을 수 있게.
        if grain_mismatch:
            cautions = cautions + [{
                "code": "GRAIN_MISMATCH_WITH_SQL",
                "source": "sql_integrity",
                "severity": "high",
                "message_ko": (
                    f"SQL이 선언한 grain('{declared_grain}')과 EDA가 추정한 데이터 단위"
                    f"('{data_level.get('grain_hint', '')}')가 일치하지 않습니다. 집계 단위 해석에 주의가 필요합니다."
                ),
                "recommended_action": ["verify_grain_before_group_analysis"],
            }]
        # 상류 원천 테이블의 GE 정합성 이슈(스코핑+fail_only+100줄 캡, #123 함수 재사용). 없으면 "".
        integrity_text = load_scoped_integrity_text(state.get("plan_source_tables") or [])
        if integrity_text:
            fail_tables = _fail_tables_from_integrity_text(integrity_text)
            tbls = ", ".join(fail_tables) if fail_tables else "원천 테이블"
            cautions = cautions + [{
                "code": "UPSTREAM_INTEGRITY_ISSUE",
                "source": "sql_integrity",
                "severity": "medium",
                "message_ko": f"원천 테이블({tbls})에서 정합성 이슈가 확인되었습니다.",
                "recommended_action": ["treat_as_interpretation_limit", "report_conclusions_conservatively"],
                "details": integrity_text,  # 원문은 요약과 분리해 details 로(프롬프트/UI가 필요시 참조)
            }]

        # clustering_result 가 비어있으면(컨트롤러가 안 돌린 경우) skip 처리
        clustering = state.get("clustering_result") or {}
        statistical_metadata = {
            "row_count":          len(df),
            "data_level":         data_level,
            "sample_reliability": sample_reliability,
            "cautions":           cautions,
            "analysis_constraints": analysis_constraints,
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

        # codegen 탈출구 성공 결과를 수치 계약에 편입(도구로 답 못 낸 질문에만 채워짐).
        # 생성코드 원문(provenance)·계측·llm_generated caution 포함. 실패(out_of_domain)는
        # 여기 안 넣고 agent.py payload의 top-level 플래그로 노출한다.
        codegen = state.get("codegen") or {}
        if codegen.get("status") == "success":
            statistical_metadata["adhoc_analysis"] = codegen

    all_results = f"""
[구조 탐색] {state.get('inspect_result', '해당 없음')}
[품질 점검] {state.get('quality_result', '해당 없음')}
[분포 분석] {state.get('distribution_result', '해당 없음')}
[그룹 비교] {state.get('comparison_result', '해당 없음')}
[관계 탐색] {state.get('relationship_result', '해당 없음')}
[시간 분석] {state.get('time_result', '해당 없음')}
[클러스터링] {json.dumps(statistical_metadata.get('clustering', {}), ensure_ascii=False, default=str, separators=(',', ':'))}
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

    # 서술 뒤의 짧은 facts JSON만 분리한다. cautions는 위에서 계산한 결정론적 규칙 결과를 쓴다.
    insight_result, facts = split_marked_json(insight_result, SUMMARY_FACTS_MARKER)
    summary_facts = [str(f).strip() for f in facts if str(f).strip()][:6] if isinstance(facts, list) else []

    # 분석 노드들이 ctx에 누적한 차트 주문서를 state로 노출 + 아티팩트로 영속화.
    chart_requests = list(get_context().chart_requests)
    _persist_chart_requests(chart_requests)

    return {
        "insight_result": insight_result,
        "summary_facts": summary_facts,
        "statistical_metadata": statistical_metadata,
        "data_level": data_level,
        "cautions": cautions,
        "analysis_constraints": analysis_constraints,
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

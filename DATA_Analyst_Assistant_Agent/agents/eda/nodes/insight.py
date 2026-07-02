"""insight 노드 — downstream용 statistical_metadata 집계 + LLM 인사이트/구조해석."""

from __future__ import annotations

import json
import math
from typing import Any, Dict

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import append_errors, get_context, get_llm, safe_json_parse
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.tool_runner import run_node_with_retry
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import insight_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState

try:  # 제공자에 따라 openai 예외가 없을 수 있어 방어적으로 import
    from openai import RateLimitError
except Exception:  # noqa: BLE001
    class RateLimitError(Exception):
        pass


# ─────────────────────────────
# LLM 탐지 레이어 (열린 주의사항 탐지) — soft caution only, hard 제약 X
# rule(reliability.py)이 못 잡는 문제(클래스 불균형·계절성 등)를 계산된 숫자에 근거해 추가한다.
# ─────────────────────────────
_LLM_CAUTION_MAX = 3
_ALLOWED_SEVERITY = {"low", "medium", "high"}


def _llm_infer_cautions(numeric_summary: Dict[str, Any], user_question: str, existing_codes: set) -> list:
    """계산된 통계를 LLM에 보여주고 rule이 놓친 추가 주의사항을 탐지(source=llm_inferred).

    가드레일: 스키마 검증 · evidence_keys 필수 · rule code 중복 제거 · 최대 3개 ·
    blocked_operations/constraint_ids 절대 생성 안 함(hard contract = rule only).
    실패/토큰 불가 시 [] 폴백(rule cautions는 그대로 유지).
    """
    try:
        prompt = (
            "너는 EDA 결과를 감사하는 데이터 품질 점검자다. 아래 '계산된 통계 요약'과 사용자 질문을 보고, "
            "이미 발견된 주의사항 외에 분석 에이전트가 놓치면 안 될 추가 주의사항을 찾아라.\n"
            "규칙:\n"
            "- 통계에 실제로 근거가 있는 것만. 근거 없으면 만들지 마라(빈 리스트 허용).\n"
            f"- 이미 있는 code는 다시 만들지 마라: {sorted(existing_codes)}\n"
            "- data_level.is_aggregated가 true면 집계본이다: group_comparison의 min_group_n/max_group_n으로 "
            "'표본 부족' 경고를 만들지 마라(그룹당 1행이라 1이 당연). 표본 신뢰도는 sample_reliability.low_n_groups를 근거로 하라.\n"
            "- 예시 유형: 클래스 불균형, 시계열 계절성, 이중분포(bimodal), 절단/검열, 결측 편중.\n"
            "- 각 항목은 soft 경고다. 분석을 '금지'하지 마라(권고까지만).\n"
            f"- 최대 {_LLM_CAUTION_MAX}개.\n\n"
            f"[사용자 질문]\n{user_question}\n\n"
            f"[계산된 통계 요약]\n{json.dumps(numeric_summary, ensure_ascii=False, default=str)[:4000]}\n\n"
            "아래 JSON 배열만 출력하라(설명 금지). 각 원소:\n"
            '{"code":"UPPER_SNAKE","severity":"low|medium|high","message_ko":"한국어 설명",'
            '"recommended_action":["english_tag"],"evidence_keys":["어느 통계를 봤는지"]}'
        )
        parsed = safe_json_parse(get_llm().invoke(prompt).content, [])
    except Exception:  # noqa: BLE001
        return []

    if not isinstance(parsed, list):
        return []

    out, seen = [], set(existing_codes)
    for item in parsed:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", "")).strip().upper()
        msg = str(item.get("message_ko", "")).strip()
        ev = item.get("evidence_keys")
        # 가드레일: code·message_ko·evidence_keys 필수 + code 중복 금지
        if not code or not msg or not ev or code in seen:
            continue
        sev = item.get("severity", "low")
        if sev not in _ALLOWED_SEVERITY:
            sev = "low"
        action = item.get("recommended_action") or []
        action = action if isinstance(action, list) else [action]
        ev = ev if isinstance(ev, list) else [ev]
        out.append({
            "code": code,
            "source": "llm_inferred",                 # 강제(신뢰도 구분용)
            "severity": sev,
            "message_ko": msg,
            "recommended_action": [str(a) for a in action][:5],
            "evidence_keys": [str(e) for e in ev][:5],
            # blocked_operations/constraint_ids 안 붙임 — hard 제약은 rule만
        })
        seen.add(code)
        if len(out) >= _LLM_CAUTION_MAX:
            break
    return out


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
            pear = df[numeric_cols].corr()
            spear = df[numeric_cols].corr(method="spearman")
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    a, b = numeric_cols[i], numeric_cols[j]
                    p = float(pear.iloc[i, j])
                    sp = float(spear.iloc[i, j])
                    gap = abs(sp) - abs(p)                    # 비선형 신호(단조인데 비선형)
                    nonlin = ("weak" if abs(p) < 0.1 and abs(sp) < 0.1 else
                              "monotonic_nonlinear" if gap > 0.15 else "linear")
                    pair_df = df[[a, b]].dropna()
                    npts = len(pair_df)
                    entry = {
                        "pearson_r":    _r(p, 3),
                        "spearman_r":   _r(sp, 3),
                        "nonlinearity": nonlin,
                        "n":            npts,
                    }
                    # 관계 있는 쌍만 산점도 모양(binned_trend) 등 추가 (약한 쌍엔 bloat 방지)
                    if npts >= 30 and (abs(p) >= 0.2 or abs(sp) >= 0.2):
                        entry["r_squared_linear"] = _r(p * p, 3)   # 선형 fit 설명력임을 명시
                        try:
                            x = pair_df[a].astype(float)
                            y = pair_df[b].astype(float)
                            bins = pd.qcut(x, q=min(10, max(2, x.nunique())), duplicates="drop")
                            bt = []
                            for interval, grp in y.groupby(bins, observed=True):
                                bt.append({
                                    "x_range":  [_r(interval.left, 2), _r(interval.right, 2)],
                                    "y_median": _r(grp.median(), 3),
                                    "y_iqr":    [_r(grp.quantile(0.25), 3), _r(grp.quantile(0.75), 3)],
                                    "n":        int(len(grp)),
                                })
                            entry["binned_trend"] = bt          # x구간별 y중앙값+IQR = 산점도 압축
                            entry["binning"] = {"method": "quantile", "n_bins": len(bt)}  # 구간 생성 기준
                        except Exception:  # noqa: BLE001
                            pass
                    corr_pairs[f"corr_{a}_vs_{b}"] = entry      # 키 형식 유지(소비처 호환)

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
                    g   = df.groupby(key_col)[col]
                    grp = g.mean().dropna()                        # 그룹별 평균
                    if grp.empty:
                        continue
                    counts = g.count().reindex(grp.index)          # 그룹별 표본 수(col 기준 non-NaN)
                    group_mean = float(grp.mean())
                    group_max  = float(grp.max())
                    group_min  = float(grp.min())
                    group_std  = float(grp.std())                  # 그룹 평균들의 표준편차(그룹 1개면 NaN)
                    max_n      = int(counts.max())

                    entry = {
                        "top3_groups":    {str(k): round(float(v), 4) for k, v in grp.nlargest(3).items()},
                        "bottom3_groups": {str(k): round(float(v), 4) for k, v in grp.nsmallest(3).items()},
                        "group_max":      round(group_max, 4),
                        "group_min":      round(group_min, 4),
                        "group_std":      _r(group_std, 4),
                        "n_groups":       int(len(grp)),           # 이하 신규 — 효과크기 해석 맥락
                        "min_group_n":    int(counts.min()),
                        "max_group_n":    max_n,
                    }

                    # 효과크기 eta² = 그룹이 이 변수 분산을 몇 % 설명하나(SS_between/SS_total).
                    # 그룹당 복수 관측(raw)일 때만 의미 있음 — 집계본(그룹당 1행)이면 스킵.
                    if max_n >= 2:
                        grand      = float(df[col].mean())
                        ss_total   = float(((df[col] - grand) ** 2).sum())
                        ss_between = float((counts * (grp - grand) ** 2).sum())
                        eta = ss_between / ss_total if ss_total > 0 else None
                        entry["eta_squared"] = _r(eta, 4)
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

                    # 그룹 간 격차 배율 — 최저 그룹 평균이 양수일 때만(0/음수면 무의미)
                    entry["spread_ratio"] = _r(group_max / group_min, 4) if group_min > 0 else None
                    # 그룹 평균들의 변동계수 — 전체 평균이 0 근처면 폭발하므로 스킵
                    entry["cv_across_groups"] = _r(group_std / group_mean, 4) if abs(group_mean) > 1e-9 else None

                    group_comparison[col] = entry
                except Exception:
                    pass

        # 데이터 한계 자가점검 (코드, LLM 없음): 원본/집계 판정 + 표본 신뢰도 + 주의사항
        data_level = detect_data_level(df, key_col=key_col, numeric_cols=numeric_cols)
        sample_reliability = assess_sample_reliability(
            df, key_col=key_col, count_col=count_col, data_level=data_level.get("level", "unknown"))

        # 집계본의 group key는 ID가 아니라 '묶는 기준' — id_like 오판 교정(집계본은 그룹당 1행이라
        # key_col이 전부 유니크→ID로 오인됨). semantic_type 전체 체계는 후속(커밋7), 여긴 이 케이스만.
        if data_level.get("is_aggregated") and key_col and isinstance(dist_stats.get(key_col), dict):
            gk = dist_stats[key_col]
            gk["semantic_type"] = "group_key"
            gk["is_id_like"] = False
            if isinstance(gk.get("eda_notes"), dict):
                gk["eda_notes"]["recommended_handling"] = ["use_as_group_key"]

        # rule 기반(결정론): 구조체 cautions + hard 계약(analysis_constraints)
        cautions = build_cautions(data_level, sample_reliability, corr_pairs, distribution=dist_stats)
        analysis_constraints = build_analysis_constraints(data_level)

        # LLM 천장(soft 탐지): rule이 못 잡은 추가 주의사항. 실패해도 rule cautions는 유지.
        numeric_summary = {
            "data_level":        {"is_aggregated": data_level.get("is_aggregated"),
                                  "grain_hint": data_level.get("grain_hint")},
            "sample_reliability": sample_reliability,   # 표본 신뢰도의 올바른 근거(min_group_n 아님)
            "distribution":      dist_stats,
            "group_comparison":  group_comparison,
            "correlation_pairs": corr_pairs,
            "time_result":       state.get("time_result"),
        }
        cautions = cautions + _llm_infer_cautions(
            numeric_summary, state["user_question"], {c["code"] for c in cautions})

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

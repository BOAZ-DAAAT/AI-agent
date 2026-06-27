"""데이터 한계 자가점검 (LLM 없음, 순수 코드).

EDA가 받은 데이터를 스스로 진단한다:
  1) detect_data_level     — 원본(개별 관측)인지 집계본인지 + grain 추정
  2) assess_sample_reliability — 표본 적은 그룹 = 비교 신뢰도 낮음 표시
  3) build_cautions        — 위 진단 + 정적 주의사항을 사람이 읽을 문장 리스트로

스펙(노션 EDA Agent #3·#14·#15)의 "분석 단위/원본·집계 판단 + 표본 신뢰도 + 해석 주의사항"
중 grain·LLM에 의존하지 않고 지금 데이터로 바로 되는 부분만 가져온 것.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# 그룹 간 차이 검정(그룹당 복수 관측이 반드시 필요) — 집계본(그룹당 1행)에선 수학적으로 불가
_GROUP_DIFF_METHODS = ("anova", "분산분석", "t검정", "t-검정", "t test", "ttest", "tukey", "튜키")

# 집계 산출물임을 시사하는 컬럼명 힌트
_AGG_NAME_HINTS = ("avg_", "mean_", "total_", "sum_", "count_", "cnt_",
                   "_avg", "_mean", "_total", "_sum", "_count", "_cnt", "_rate", "_ratio")

LOW_SAMPLE_THRESHOLD = 30  # 그룹 표본 신뢰도 경고 기준 (통상적 최소 표본)


def _agg_named_columns(cols: List[str]) -> List[str]:
    return [c for c in cols if any(h in c.lower() for h in _AGG_NAME_HINTS)]


def detect_data_level(df: pd.DataFrame, key_col: Optional[str] = None,
                      numeric_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """원본(raw) vs 집계(aggregated) 판정 + grain 추정.

    신호 두 가지를 본다:
      - key_col이 행과 1:1이면(그룹당 1행) 집계본 가능성
      - 컬럼명이 avg_/total_/_rate 등 집계산출물 형태면 집계본 가능성
    둘 중 하나라도 강하면 aggregated, key당 여러 행이고 집계명 없으면 raw.
    """
    n_rows = len(df)
    result: Dict[str, Any] = {"level": "unknown", "n_rows": n_rows, "grain_hint": "", "reason": "",
                              "confidence": "low"}

    agg_cols = _agg_named_columns(numeric_cols or list(df.columns))
    one_row_per_key = False
    if key_col and key_col in df.columns:
        per_key = df.groupby(key_col).size()
        med = float(per_key.median())
        n_keys = int(df[key_col].nunique())
        result["rows_per_key_median"] = round(med, 2)
        result["n_keys"] = n_keys
        one_row_per_key = (n_keys == n_rows) or (med <= 1.0)

    if one_row_per_key and key_col:
        result.update(level="aggregated", grain_hint=f"one row per {key_col}",
                      confidence="high",
                      reason=f"{key_col}가 행과 1:1 → 집계 단위(그룹당 1행)")
    elif agg_cols:
        result.update(level="aggregated", confidence="medium",
                      grain_hint=result.get("grain_hint", ""),
                      reason=f"집계형 컬럼명 존재: {agg_cols[:4]}")
    elif key_col and result.get("rows_per_key_median", 0) >= 2:
        result.update(level="raw", grain_hint=f"multiple rows per {key_col}",
                      confidence="medium",
                      reason=f"{key_col} 그룹당 중앙 {result['rows_per_key_median']:.0f}행 → 개별 관측")
    else:
        result["reason"] = "판정 신호 부족(key_col 없음/집계명 없음) — 보류"

    result["aggregated_named_columns"] = agg_cols
    return result


def assess_sample_reliability(df: pd.DataFrame, key_col: Optional[str] = None,
                              count_col: Optional[str] = None, data_level: str = "unknown",
                              min_n: int = LOW_SAMPLE_THRESHOLD) -> Dict[str, Any]:
    """표본 적은 그룹을 찾는다. 우선순위: count_col(볼륨) 합 → 원본이면 그룹 행수 → count_col 단독."""
    out: Dict[str, Any] = {"threshold": min_n, "basis": "", "low_n_groups": [],
                           "low_n_count": 0, "total_groups": 0, "note": ""}

    sizes = None
    if count_col and count_col in df.columns and key_col and key_col in df.columns:
        sizes = df.groupby(key_col)[count_col].sum()
        out["basis"] = f"{count_col} 합(그룹별)"
    elif count_col and count_col in df.columns:
        sizes = df[count_col]
        out["basis"] = f"{count_col}(행별)"
    elif data_level == "raw" and key_col and key_col in df.columns:
        sizes = df.groupby(key_col).size()
        out["basis"] = "그룹 행수"

    if sizes is None or len(sizes) == 0:
        out["note"] = "표본 수 기준 컬럼/그룹 없음 — 신뢰도 점검 보류"
        return out

    low = sizes[sizes < min_n].sort_values()
    out["total_groups"] = int(len(sizes))
    out["low_n_count"] = int(len(low))
    out["low_n_groups"] = [{"group": str(k), "n": int(v)} for k, v in low.head(10).items()]
    if out["low_n_count"]:
        out["note"] = f"{out['total_groups']}개 중 {out['low_n_count']}개 그룹이 표본 {min_n} 미만 → 해당 그룹 비교 신뢰도 낮음"
    else:
        out["note"] = f"모든 그룹 표본 {min_n} 이상 → 표본 신뢰도 양호"
    return out


def correct_hypothesis_feasibility(df: pd.DataFrame, hypotheses_text: str,
                                   data_level: Optional[dict] = None) -> Tuple[str, List[dict]]:
    """가설 텍스트에서 '불가능한 검정'을 코드로(사실 기준) 교정한다 — LLM 아님, 영구적.

    그룹 간 차이 검정(ANOVA/t검정)은 그룹당 복수 관측이 있어야 한다. feature 그룹이
    사실상 그룹당 1행(집계본)이면 그 검정은 못 돌리므로, 해당 가설의 '현재데이터:' 라벨을
    '추가 필요(원본 행 필요)'로 강제 교정한다. (의견 아니라 groupby 행수라는 사실로 판정)
    반환: (교정된 텍스트, 교정 내역).
    """
    if df is None or not hypotheses_text:
        return hypotheses_text, []

    aggregated = (data_level or {}).get("level") == "aggregated"

    def _resolve_col(feature: Optional[str]) -> Optional[str]:
        """LLM이 컬럼명을 줄여 쓸 수 있어(category→product_category) 퍼지 매칭으로 실제 컬럼 찾기."""
        if not feature:
            return None
        if feature in df.columns:
            return feature
        fl = feature.lower()
        cands = [c for c in df.columns if fl in c.lower() or c.lower() in fl]
        obj = [c for c in cands if df[c].dtype == object]  # 그룹 컬럼은 보통 범주형
        return (obj or cands or [None])[0]

    def _group_diff_feasible(feature: Optional[str]) -> Tuple[Optional[bool], Optional[str]]:
        col = _resolve_col(feature)
        if col is None:
            # 컬럼 못 찾음 — 집계본이면 그룹 간 검정은 어차피 불가(폴백)
            return (False if aggregated else None), None
        try:
            sizes = df.groupby(col).size()
        except Exception:  # noqa: BLE001
            return None, col
        # 그룹당 2행 이상인 그룹이 2개 이상이어야 그룹 간 차이 검정이 가능
        return int((sizes >= 2).sum()) >= 2, col

    lines = hypotheses_text.split("\n")
    corrections: List[dict] = []
    cur = {"method": "", "feature": None, "data_idx": None}

    def _flush() -> None:
        method = (cur["method"] or "").lower()
        if cur["data_idx"] is None or not any(k in method for k in _GROUP_DIFF_METHODS):
            return
        feasible, col = _group_diff_feasible(cur["feature"])
        if feasible is False:
            label = col or cur["feature"] or "그룹"
            lines[cur["data_idx"]] = (
                f"현재데이터: 추가 필요 (원본 행 필요 — '{label}'가 그룹당 1행이라 "
                f"{cur['method'].strip()} 불가)")
            corrections.append({"feature": cur["feature"], "resolved": col, "method": cur["method"].strip()})

    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("[가설"):
            _flush()
            cur = {"method": "", "feature": None, "data_idx": None}
        elif s.startswith("검증방법:"):
            cur["method"] = s.split(":", 1)[1]
        elif s.startswith("필요변수:"):
            m = re.search(r"feature\s*=\s*\[?\s*([A-Za-z0-9_]+)", s)
            if m:
                cur["feature"] = m.group(1)
        elif s.startswith("현재데이터:"):
            cur["data_idx"] = i
    _flush()  # 마지막 블록

    return "\n".join(lines), corrections


def build_cautions(data_level: Dict[str, Any], reliability: Dict[str, Any],
                   correlation_pairs: Optional[Dict[str, float]] = None) -> List[str]:
    """진단 결과 + 정적 주의사항을 사람이 읽을 문장 리스트로."""
    cautions: List[str] = []

    if data_level.get("level") == "aggregated":
        grain = data_level.get("grain_hint") or "집계 단위"
        cautions.append(
            f"집계 데이터({grain})로 판단됨 → 개별 관측 수준 분석(분포 형태·회귀·이상치)은 부적합하고, "
            "집계 단위 비교·추세 해석에 적합."
        )
    elif data_level.get("level") == "raw":
        cautions.append("개별 관측(raw) 데이터로 판단됨 → 분포·회귀·그룹 분포 비교 가능.")

    if reliability.get("low_n_count"):
        ex = ", ".join(f"{g['group']}(n={g['n']})" for g in reliability.get("low_n_groups", [])[:3])
        cautions.append(
            f"표본 적은 그룹 {reliability['low_n_count']}개 존재({ex} 등, 기준 {reliability.get('threshold')}) "
            "→ 해당 그룹의 평균·비교는 신뢰도 낮으니 통합/제외 고려."
        )

    if correlation_pairs and any(abs(v) >= 0.3 for v in correlation_pairs.values()):
        cautions.append("관찰된 상관은 인과가 아니며, 교란변수(카테고리·지역·가격 등) 통제가 필요할 수 있음.")

    return cautions

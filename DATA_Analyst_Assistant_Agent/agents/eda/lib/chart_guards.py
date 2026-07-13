"""퇴화 차트 가드 (LLM 없음, 순수 코드).

chart_selector 1.5단계에서, 이미 계산된 통계 숫자(statistical_metadata)로 '정보가 없는'
차트를 사후 컷한다. 기존 _drop_weak_scatters(약한 상관 산점도 제거)와 같은 계열의 결정론
가드레일을, 분포/그룹비교 신호로 확장한 것.

컷 대상(보수적 — 신호가 명백할 때만 컷, 신호 없거나 매칭 실패 시 유지):
  1) catdist_{col}              : 범주 1개(unique<=1) 또는 한 범주 >=95% 지배  → 막대가 무의미
  2) dist_/box_/violin_{col}    : 수치 상수(std≈0 또는 unique<=1)             → 분포 그림이 무의미
  3) bar_top_/bar_bottom_{metric}: 그룹 1개 또는 그룹 평균 전부 동일           → 순위 막대가 무의미
  4) heatmap_matrix             : 모든 지표가 그룹 간 평탄(분산 0)             → 히트맵 전체가 무의미

차트 파일명은 visualize.py 가 실제 컬럼/지표명으로 생성하므로 distribution/group_comparison
키와 정확히 일치한다(퍼지매칭 불필요). 매칭 실패 시엔 컷하지 않는다(Phase 1 과 같은 보수 원칙).

보류(지금 stat 만으론 신뢰성 있게 판정 불가):
  - 시계열 ts_/multiline_{metric} 의 '유효 시점 <3': time_result 에 시점 개수가 없고 stat 에도
    안 실려 df 없이는 못 셈 → df 를 넘기게 되면 확장.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

DOMINANT_CATEGORY_SHARE = 0.95   # 한 범주가 이 비율 이상 차지 → 사실상 단일 범주
CONSTANT_STD_EPS = 1e-9          # 수치 표준편차가 이 값 이하 → 상수


def _basename_noext(path: str) -> str:
    b = os.path.basename(path)
    return b[:-4] if b.lower().endswith(".png") else b


def _num(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _is_flat_group(g: Dict[str, Any]) -> bool:
    """그룹 비교가 평탄한가(그룹 1개거나 그룹 평균이 전부 동일)."""
    ng = g.get("n_groups")
    if ng is not None and ng <= 1:
        return True
    gmax, gmin = _num(g.get("group_max")), _num(g.get("group_min"))
    if gmax is not None and gmin is not None and gmax == gmin:
        return True
    std = _num(g.get("group_std"))
    return std is not None and std <= CONSTANT_STD_EPS


def _degenerate_reason(name: str, distribution: Dict[str, Any],
                       group_comparison: Dict[str, Any]) -> Optional[str]:
    """차트 파일명(확장자 제거) → 퇴화 사유. 컷 안 하면 None."""
    # 1) 범주 빈도 막대
    if name.startswith("catdist_"):
        d = distribution.get(name[len("catdist_"):])
        if isinstance(d, dict) and d.get("type") == "categorical":
            uc = d.get("unique_count")
            if uc is not None and uc <= 1:
                return f"단일 범주(unique_count={uc})"
            share = _num(d.get("top1_share"))
            if share is not None and share >= DOMINANT_CATEGORY_SHARE:
                return f"한 범주 지배(top1_share={share})"
        return None

    # 2) 수치 분포(히스토그램/박스/바이올린)
    for pref in ("dist_", "box_", "violin_"):
        if name.startswith(pref):
            d = distribution.get(name[len(pref):])
            if isinstance(d, dict) and d.get("type") == "numeric":
                uc = d.get("unique_count")
                if uc is not None and uc <= 1:
                    return f"상수 분포(unique_count={uc})"
                std = _num(d.get("std"))
                if std is not None and std <= CONSTANT_STD_EPS:
                    return "상수 분포(std≈0)"
            return None

    # 3) 순위 막대(그룹별 지표 평균)
    for pref in ("bar_top_", "bar_bottom_"):
        if name.startswith(pref):
            g = group_comparison.get(name[len(pref):])
            if isinstance(g, dict) and _is_flat_group(g):
                return "그룹 간 평탄(순위 무의미)"
            return None

    # 4) 카테고리×지표 히트맵 — 모든 지표가 평탄하면 전체가 무의미
    if name == "heatmap_matrix":
        metrics = [g for g in group_comparison.values() if isinstance(g, dict)]
        if metrics and all(_is_flat_group(g) for g in metrics):
            return "모든 지표가 그룹 간 평탄"
        return None

    return None


def drop_degenerate_charts(paths: List[str],
                           statistical_metadata: Optional[Dict[str, Any]]
                           ) -> Tuple[List[str], List[Dict[str, str]]]:
    """퇴화 차트를 컷한다 → (유지 경로 리스트, 드롭 메타 리스트).

    드롭 메타: [{"chart": 파일명, "reason": 사유}] — 설명가능성/디버깅용.
    신호가 없거나 매칭 실패면 컷하지 않는다(보수적).
    """
    stat = statistical_metadata or {}
    distribution = stat.get("distribution", {}) or {}
    group_comparison = stat.get("group_comparison", {}) or {}

    kept: List[str] = []
    dropped: List[Dict[str, str]] = []
    for p in paths:
        reason = _degenerate_reason(_basename_noext(p), distribution, group_comparison)
        if reason:
            dropped.append({"chart": os.path.basename(p), "reason": reason})
        else:
            kept.append(p)
    return kept, dropped

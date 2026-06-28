"""plot_* 결과 → ChartRequest(차트 '주문서') 변환 빌더.

EDA가 발행하는 차트 주문서를 만든다. matplotlib/LLM/encoding 결정은 하지 않고,
intent(자연어 의도) + stats(검증된 수치) + columns(사용 컬럼) + hint(표준 차트 종류)만 담는다.
Phase B의 chart/ 패키지가 이 주문서를 Vega-Lite로 렌더한다.

각 빌더는 해당 skill의 반환 dict를 받아 ChartRequest dict 리스트를 돌려준다.
누적/중복제거는 호출 측(react 툴·노드 → _runtime.accumulate_chart_requests)이 담당한다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from DATA_Analyst_Assistant_Agent.chart.contract import ChartRequest


def _req(intent: str, columns: Dict[str, Dict[str, Any]], stats: Dict[str, Any],
         hint: Optional[str] = None) -> dict:
    return ChartRequest(intent=intent, columns=columns, stats=stats, hint=hint).model_dump()


# ─────────────────────────────
# distribution skill
# ─────────────────────────────
def from_distribution_skill(result: dict) -> List[dict]:
    """수치형 분포(히스토그램/바이올린/박스 중 생성된 것 1종) + 범주형 빈도."""
    out: List[dict] = []

    # 수치형: 같은 컬럼에 중복 발행하지 않도록 가용한 표현 1종만 선택
    if result.get("distributions"):
        numeric, hint, kind = result["distributions"], "histogram", "히스토그램"
    elif result.get("violins"):
        numeric, hint, kind = result["violins"], "violin", "바이올린"
    elif result.get("boxplots"):
        numeric, hint, kind = result["boxplots"], "boxplot", "박스플롯"
    else:
        numeric, hint, kind = {}, None, None

    for col, st in (numeric.get("stats") or {}).items():
        out.append(_req(f"{col}의 분포를 {kind}으로 보여줘", {col: {"type": "numeric"}}, st, hint))

    for col, st in ((result.get("category_distribution") or {}).get("stats") or {}).items():
        out.append(_req(f"{col}의 범주별 빈도 분포(상위 항목)", {col: {"type": "categorical"}}, st, "bar"))

    # 다변수 교차(원본 데이터에서만 발화)
    gb = result.get("grouped_box") or {}
    if gb.get("stats"):
        out += from_grouped_box(gb, gb.get("key_col"))
    dbt = result.get("dist_by_target") or {}
    if dbt.get("stats"):
        out += from_distribution_by_target(dbt)

    return out


# ─────────────────────────────
# relationship skill
# ─────────────────────────────
def from_relationship_skill(result: dict) -> List[dict]:
    """상관 행렬 히트맵 + 강한 쌍의 산점도."""
    out: List[dict] = []

    cstats = (result.get("correlation") or {}).get("stats") or {}
    matrix = cstats.get("correlation_matrix") or {}
    if matrix:
        cols = {c: {"type": "numeric"} for c in matrix.keys()}
        out.append(_req("수치형 변수 간 상관관계 행렬을 한눈에 보여줘",
                        cols, {"strong_pairs": cstats.get("strong_pairs", [])}, "heatmap"))

    for pair, st in ((result.get("scatter_pairs") or {}).get("stats") or {}).items():
        if " vs " not in pair:
            continue
        a, b = pair.split(" vs ", 1)
        r = st.get("pearson_r")
        direction = "양" if (r or 0) >= 0 else "음"
        out.append(_req(f"{a}와 {b}의 관계를 산점도로({direction}의 상관 강조)",
                        {a: {"type": "numeric"}, b: {"type": "numeric"}}, st, "scatter"))

    dbt = result.get("dist_by_target") or {}
    if dbt.get("stats"):
        out += from_distribution_by_target(dbt)

    return out


# ─────────────────────────────
# comparison skill
# ─────────────────────────────
def from_comparison_skill(result: dict, key_col: Optional[str] = None,
                          measure_cols: Optional[list] = None) -> List[dict]:
    """카테고리×지표 종합 비교 차트(막대/히트맵/그룹막대/레이더/버블)."""
    out: List[dict] = []
    kc = key_col or "category"
    cat_col = {kc: {"type": "categorical"}}
    num_cols = {m: {"type": "numeric"} for m in (measure_cols or [])}

    for metric, st in ((result.get("top_n_barplot") or {}).get("stats") or {}).items():
        out.append(_req(f"{kc}별 {metric} 상위/하위 비교(막대)",
                        {kc: {"type": "categorical"}, metric: {"type": "numeric"}}, st, "bar"))

    if (result.get("heatmap_matrix") or {}).get("stats"):
        out.append(_req(f"{kc}×지표 정규화 히트맵으로 종합 비교",
                        {**cat_col, **num_cols}, result["heatmap_matrix"]["stats"], "heatmap"))

    if (result.get("grouped_bar") or {}).get("stats"):
        out.append(_req(f"{kc}별 지표 그룹막대로 종합 비교",
                        {**cat_col, **num_cols}, result["grouped_bar"]["stats"], "grouped_bar"))

    if (result.get("radar") or {}).get("stats"):
        out.append(_req(f"상위 {kc}의 지표 프로파일을 레이더로 비교",
                        {**cat_col, **num_cols}, result["radar"]["stats"], "radar"))

    bubble_stats = (result.get("bubble") or {}).get("stats") or {}
    axes = bubble_stats.get("axes") or {}
    if axes:
        cols = {kc: {"type": "categorical"}}
        for v in axes.values():
            if v:
                cols[v] = {"type": "numeric"}
        out.append(_req(f"{kc}의 다지표 포지셔닝을 버블차트로", cols, bubble_stats, "scatter"))

    ct = result.get("crosstab") or {}
    if ct.get("stats"):
        out += from_crosstab(ct)

    return out


# ─────────────────────────────
# time skill
# ─────────────────────────────
def from_time_skill(result: dict, time_cols: Optional[list] = None) -> List[dict]:
    """시계열 추세(선) + 시즌성(막대)."""
    out: List[dict] = []
    tcol = (time_cols or ["time"])[0] if time_cols else "time"

    for metric, st in ((result.get("timeseries") or {}).get("stats") or {}).items():
        out.append(_req(f"{metric}의 시간 추세를 선그래프로",
                        {tcol: {"type": "temporal"}, metric: {"type": "numeric"}}, st, "line"))

    for key, st in ((result.get("seasonality") or {}).get("stats") or {}).items():
        out.append(_req(f"{key} 기준 시즌성을 막대로", {tcol: {"type": "temporal"}}, st, "bar"))

    ml = result.get("multiline") or {}
    if ml.get("stats"):
        out += from_multiline(ml)

    return out


# ─────────────────────────────
# 그룹별 분포 (범주 × 수치, 다변수 교차)
# ─────────────────────────────
def from_grouped_box(result: dict, key_col: str) -> List[dict]:
    """카테고리별 수치 분포(그룹 박스플롯) 주문서. 원본 데이터에서만 발행된다."""
    out: List[dict] = []
    kc = key_col or "category"
    for metric, st in (result.get("stats") or {}).items():
        out.append(_req(
            f"{kc}별 {metric} 분포를 그룹 박스플롯으로 비교(평균이 숨기는 산포·이상치 확인)",
            {kc: {"type": "categorical"}, metric: {"type": "numeric"}},
            st, "grouped_box",
        ))
    return out


def from_distribution_by_target(result: dict) -> List[dict]:
    """target 수준별 수치 분포(쿼리 정조준) 주문서."""
    out: List[dict] = []
    target = result.get("target") or "target"
    for metric, st in (result.get("stats") or {}).items():
        out.append(_req(
            f"{target} 수준별 {metric} 분포를 비교해 {metric}이 {target}에 영향을 주는지 본다",
            {target: {"type": "ordinal_or_numeric"}, metric: {"type": "numeric"}},
            st, "grouped_box",
        ))
    return out


def from_multiline(result: dict) -> List[dict]:
    """카테고리별 시간 추세(다중 선) 주문서."""
    out: List[dict] = []
    kc = result.get("key_col") or "category"
    tc = result.get("time_col") or "time"
    for metric, st in (result.get("stats") or {}).items():
        out.append(_req(
            f"{kc}별 {metric}의 시간 추세를 여러 선으로 비교(누가 언제 뜨고 지나)",
            {tc: {"type": "temporal"}, kc: {"type": "categorical"}, metric: {"type": "numeric"}},
            st, "multiline",
        ))
    return out


def from_crosstab(result: dict) -> List[dict]:
    """두 범주 교차 빈도 히트맵 주문서."""
    st = result.get("stats") or {}
    if not st:
        return []
    a, b = st.get("cat_a", "a"), st.get("cat_b", "b")
    return [_req(
        f"{a} × {b} 교차 빈도를 히트맵으로(어떤 조합이 몰리나)",
        {a: {"type": "categorical"}, b: {"type": "categorical"}},
        st, "crosstab",
    )]


# ─────────────────────────────
# clustering skill
# ─────────────────────────────
def from_clustering_skill(result: dict) -> List[dict]:
    """군집 프로파일 히트맵."""
    if not result or result.get("skip"):
        return []
    centroids = result.get("cluster_centroids") or {}
    measures: set = set()
    for metrics in centroids.values():
        if isinstance(metrics, dict):
            measures.update(metrics.keys())
    cols = {m: {"type": "numeric"} for m in measures}
    return [_req(
        f"{result.get('n_clusters')}개 군집의 지표 프로파일을 히트맵으로",
        cols,
        {
            "n_clusters": result.get("n_clusters"),
            "silhouette_score": result.get("silhouette_score"),
            "cluster_centroids": centroids,
        },
        "heatmap",
    )]

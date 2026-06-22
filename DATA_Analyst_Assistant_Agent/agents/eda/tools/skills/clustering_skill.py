import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from DATA_Analyst_Assistant_Agent.agents.eda.tools.chart_requests import from_clustering_skill
from DATA_Analyst_Assistant_Agent.agents.eda.tools.visualize import plot_cluster_profile, plot_cluster_scatter


def _select_k(X_scaled: np.ndarray, k_range: range) -> int:
    """실루엣 점수 기반 최적 k 선택 (데이터가 적으면 3 고정)."""
    if X_scaled.shape[0] < 10:
        return 2
    best_k, best_score = 3, -1
    for k in k_range:
        if k >= X_scaled.shape[0]:
            break
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def run_clustering_skill(
    df: pd.DataFrame,
    measure_cols: list = None,
    key_col: str = None,
    question_type: str = "",
) -> dict:
    """
    K-means 클러스터링 skill.
    comparison / distribution 타입에서만 실행.

    반환:
        cluster_labels   : {key_col값: cluster_id} (key_col 있을 때)
        cluster_centroids: {cluster_id: {measure: z-score mean}}
        n_clusters       : k
        chart_paths      : 생성된 차트 경로 목록
        skip             : 실행 생략 여부
    """
    qt = question_type.lower()

    # relationship / time 타입은 클러스터링 생략
    if qt in ("relationship", "time"):
        return {"skip": True, "reason": f"question_type={qt}에서 클러스터링 생략"}

    if df is None or df.empty:
        return {"skip": True, "reason": "데이터 없음"}

    cols = [c for c in (measure_cols or []) if c in df.columns]
    if len(cols) < 2:
        return {"skip": True, "reason": "수치형 컬럼 2개 미만 — 클러스터링 불가"}

    df_clean = df[cols].dropna()
    if len(df_clean) < 6:
        return {"skip": True, "reason": "유효 행 수 부족 (6행 미만)"}

    # 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    # 최적 k 탐색 (2~5)
    k = _select_k(X_scaled, range(2, min(6, len(df_clean))))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    silhouette = round(float(silhouette_score(X_scaled, labels)), 4) if k >= 2 else 0.0

    # 원본 df에 클러스터 레이블 부착
    df_labeled = df.loc[df_clean.index].copy()
    df_labeled["cluster"] = labels

    # 클러스터별 z-score 중심값 (히트맵용)
    centers_scaled = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=cols,
        index=[f"C{i}" for i in range(k)],
    ).round(2)

    # 클러스터별 원본 스케일 중심값 (해석용)
    centers_raw = df_labeled.groupby("cluster")[cols].mean().round(4)
    centers_raw.index = [f"C{i}" for i in centers_raw.index]

    # 클러스터 레이블 딕셔너리 (key_col 있을 때)
    cluster_labels_map = {}
    if key_col and key_col in df_labeled.columns:
        cluster_labels_map = df_labeled.set_index(key_col)["cluster"].astype(int).to_dict()

    # 차트 생성
    chart_paths = []

    # 1. 클러스터 프로파일 히트맵
    result_profile = plot_cluster_profile(centers_scaled, k)
    chart_paths.extend(result_profile.get("chart_paths", []))

    # 2. 클러스터 scatter (상관관계 가장 강한 두 컬럼 선택)
    if len(cols) >= 2:
        corr = df_clean.corr().abs()
        corr_arr = corr.to_numpy().copy()
        np.fill_diagonal(corr_arr, 0)
        corr_no_diag = pd.DataFrame(corr_arr, index=corr.index, columns=corr.columns)
        pair = corr_no_diag.unstack().idxmax()
        x_col, y_col = pair[0], pair[1]
        result_scatter = plot_cluster_scatter(df_labeled, x_col=x_col, y_col=y_col,
                                              cluster_col="cluster", key_col=key_col)
        chart_paths.extend(result_scatter.get("chart_paths", []))

    result = {
        "skip": False,
        "n_clusters": k,
        "silhouette_score": silhouette,
        "cluster_labels": {str(k): int(v) for k, v in cluster_labels_map.items()},
        "cluster_centroids": centers_raw.to_dict(orient="index"),
        "chart_paths": chart_paths,
    }
    result["chart_requests"] = from_clustering_skill(result)
    return result

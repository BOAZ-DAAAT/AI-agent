import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "all")
KEY_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "key")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(KEY_DIR,    exist_ok=True)


def set_output_dirs(grain_dir: str):
    """grain별 출력 폴더를 동적으로 설정한다. app.invoke() 전에 호출해야 한다."""
    global OUTPUT_DIR, KEY_DIR
    OUTPUT_DIR = os.path.join(grain_dir, "all")
    KEY_DIR    = os.path.join(grain_dir, "key")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(KEY_DIR,    exist_ok=True)

# ─────────────────────────────
# 공통 스타일 설정
# ─────────────────────────────

PALETTE_MAIN   = "#4C72B0"
PALETTE_ACCENT = "#DD8452"
PALETTE_NEG    = "#C44E52"
PALETTE_POS    = "#55A868"
PALETTE_SEQ    = "Blues"


def _get_numeric_cols(df: pd.DataFrame, measure_cols: list = None) -> list:
    """measure_cols가 있으면 그 중 실제 수치형만, 없으면 전체 수치형 컬럼 반환"""
    if measure_cols:
        return [c for c in measure_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    return list(df.select_dtypes(include=["float64", "int64"]).columns)


# 낮을수록 좋은 지표 키워드 — 정규화 시 반전 대상
_LOWER_IS_BETTER_KEYWORDS = [
    "day", "days", "time", "delay", "wait", "lead",
    "cancel", "return", "refund", "complaint", "error", "churn",
    "response_time", "delivery_time",
]


def _is_lower_better(col: str) -> bool:
    """컬럼명에 '낮을수록 좋은' 키워드가 포함되면 True"""
    col_lower = col.lower()
    return any(kw in col_lower for kw in _LOWER_IS_BETTER_KEYWORDS)


def _normalize_with_direction(sub: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    지표별 방향을 고려한 정규화 (0~1, 높을수록 좋음).
    '낮을수록 좋은' 지표는 정규화 후 1에서 뺌.
    """
    normalized = (sub - sub.min()) / (sub.max() - sub.min() + 1e-9)
    for col in numeric_cols:
        if _is_lower_better(col):
            normalized[col] = 1 - normalized[col]
    return normalized

def _apply_style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)


# ─────────────────────────────
# Distribution
# ─────────────────────────────

def plot_distributions(df: pd.DataFrame, measure_cols: list = None) -> dict:
    """수치형 컬럼 히스토그램 + 분포 통계"""
    paths = []
    stats = {}
    for col in _get_numeric_cols(df, measure_cols):
        s = df[col].dropna()
        stats[col] = {
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "skewness": round(float(s.skew()), 4),
        }
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(s, bins=25, color=PALETTE_MAIN, edgecolor="white", linewidth=0.6, alpha=0.85)
        ax.axvline(float(s.mean()),   color=PALETTE_ACCENT, linestyle="--", linewidth=1.4, label=f"mean={stats[col]['mean']}")
        ax.axvline(float(s.median()), color=PALETTE_NEG,    linestyle=":",  linewidth=1.4, label=f"median={stats[col]['median']}")
        ax.legend(fontsize=8, frameon=False)
        _apply_style(ax, f"Distribution: {col}", xlabel=col, ylabel="Count")
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"dist_{col}.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        paths.append(path)
    return {"chart_paths": paths, "stats": stats}


def plot_boxplots(df: pd.DataFrame, measure_cols: list = None) -> dict:
    """수치형 컬럼 박스플롯 + IQR 통계"""
    paths = []
    stats = {}
    for col in _get_numeric_cols(df, measure_cols):
        s = df[col].dropna()
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        stats[col] = {
            "q1": round(q1, 4),
            "median": round(float(s.median()), 4),
            "q3": round(q3, 4),
            "iqr": round(q3 - q1, 4),
            "lower_fence": round(q1 - 1.5 * (q3 - q1), 4),
            "upper_fence": round(q3 + 1.5 * (q3 - q1), 4),
        }
        fig, ax = plt.subplots(figsize=(5, 5))
        bp = ax.boxplot(
            s, vert=True, patch_artist=True,
            boxprops=dict(facecolor=PALETTE_MAIN, alpha=0.6, linewidth=1.2),
            medianprops=dict(color=PALETTE_ACCENT, linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", color=PALETTE_NEG, alpha=0.5, markersize=4),
        )
        ax.set_xticks([])
        _apply_style(ax, f"Boxplot: {col}", ylabel=col)
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"box_{col}.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        paths.append(path)
    return {"chart_paths": paths, "stats": stats}


def plot_violins(df: pd.DataFrame, measure_cols: list = None) -> dict:
    """수치형 컬럼 바이올린 플롯 — 분포 형태(봉우리 수, 밀도)를 박스플롯보다 풍부하게 표현"""
    paths = []
    stats = {}
    numeric_cols = _get_numeric_cols(df, measure_cols)
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 5:
            continue
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        stats[col] = {
            "mean":     round(float(s.mean()), 4),
            "median":   round(float(s.median()), 4),
            "std":      round(float(s.std()), 4),
            "q1":       round(q1, 4),
            "q3":       round(q3, 4),
            "iqr":      round(q3 - q1, 4),
            "skewness": round(float(s.skew()), 4),
        }
        fig, ax = plt.subplots(figsize=(5, 6))
        parts = ax.violinplot(s.values, vert=True, showmedians=True, showextrema=True)
        parts["cmedians"].set_color(PALETTE_ACCENT)
        parts["cmedians"].set_linewidth(2)
        for pc in parts["bodies"]:
            pc.set_facecolor(PALETTE_MAIN)
            pc.set_alpha(0.6)
            pc.set_edgecolor("white")
        ax.set_xticks([])
        _apply_style(ax, f"Violin: {col}", ylabel=col)
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"violin_{col}.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        paths.append(path)
    return {"chart_paths": paths, "stats": stats}


def plot_category_distribution(df: pd.DataFrame, top_n: int = 20,
                               max_cardinality: int = 50) -> dict:
    """범주형 컬럼 빈도 bar chart + 빈도 통계.
    고유값이 max_cardinality 초과인 컬럼(예: seller_id, product_id)은 스킵."""
    paths = []
    stats = {}
    for col in df.select_dtypes(include=["object"]).columns:
        # 고카디널리티 ID 컬럼은 의미 있는 빈도 분포가 없으므로 스킵
        if df[col].nunique() > max_cardinality:
            continue
        vc = df[col].value_counts().head(top_n)
        stats[col] = {
            "total_categories": int(df[col].nunique()),
            "top_categories": vc.head(5).to_dict(),
        }
        colors = sns.color_palette(PALETTE_SEQ, len(vc))[::-1]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(range(len(vc)), vc.values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels(vc.index, rotation=40, ha="right", fontsize=8)
        for bar, val in zip(bars, vc.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vc.values) * 0.01,
                    f"{val:,}", ha="center", va="bottom", fontsize=7.5)
        _apply_style(ax, f"Category Distribution: {col}", ylabel="Count")
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"catdist_{col}.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        paths.append(path)
    return {"chart_paths": paths, "stats": stats}


# ─────────────────────────────
# Comparison
# ─────────────────────────────

def plot_top_n_barplot(df: pd.DataFrame, top_n: int = 10, key_col: str = None, measure_cols: list = None) -> dict:
    """카테고리 키 기준 수치형 지표 상위/하위 N개 barplot + 실제 값"""
    paths = []
    stats = {}
    cat_cols = df.select_dtypes(include=["object"]).columns
    numeric_cols = _get_numeric_cols(df, measure_cols)

    if len(cat_cols) == 0 or len(numeric_cols) == 0:
        return {"chart_paths": [], "stats": {}}

    if key_col is None or key_col not in df.columns:
        key_col = cat_cols[0]
    for metric in numeric_cols:
        agg_df    = df[[key_col, metric]].dropna().groupby(key_col)[metric].mean().reset_index()
        sorted_df = agg_df.sort_values(metric, ascending=False)
        top_df    = sorted_df.head(top_n)
        bottom_df = sorted_df.tail(top_n).sort_values(metric, ascending=True)
        stats[metric] = {
            "top":    top_df.set_index(key_col)[metric].round(4).to_dict(),
            "bottom": bottom_df.set_index(key_col)[metric].round(4).to_dict(),
        }
        for label, subset, color in [("top", top_df, PALETTE_POS), ("bottom", bottom_df, PALETTE_NEG)]:
            # bottom이 전부 0이면 스킵
            if label == "bottom" and subset[metric].max() == 0:
                continue
            fig, ax = plt.subplots(figsize=(9, 5))
            vals  = subset[metric].values
            names = subset[key_col].values
            max_val = max(vals.max(), 1e-9)  # 0 나눔 방지
            bars  = ax.barh(range(len(names)), vals, color=color, alpha=0.82, edgecolor="white")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlim(left=0)
            for bar, val in zip(bars, vals):
                label_val = abs(val) if abs(val) > 1e-9 else 0.0  # -0.000 방지
                ax.text(bar.get_width() + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{label_val:.3f}", va="center", fontsize=8)
            _apply_style(ax, f"{label.upper()} {top_n}: {metric}", xlabel=metric)
            ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
            ax.grid(axis="y", visible=False)
            ax.spines["left"].set_visible(False)
            fig.tight_layout()
            path = os.path.join(OUTPUT_DIR, f"bar_{label}_{metric}.png")
            fig.savefig(path, bbox_inches="tight", dpi=120)
            plt.close(fig)
            paths.append(path)
    return {"chart_paths": paths, "stats": stats}


def plot_heatmap_matrix(df: pd.DataFrame, key_col: str = None, measure_cols: list = None,
                        max_rows: int = 40) -> dict:
    """카테고리 × 수치형 지표 정규화 히트맵 + 각 지표 상위 3개.
    행이 max_rows 초과면 첫 번째 measure 기준 상위 max_rows만 표시."""
    cat_cols     = list(df.select_dtypes(include=["object"]).columns)
    numeric_cols = _get_numeric_cols(df, measure_cols)

    if not cat_cols or not numeric_cols:
        return {"chart_path": None, "stats": {}}

    if key_col is None or key_col not in df.columns:
        key_col = cat_cols[0]
    sub = df[[key_col] + numeric_cols].dropna().groupby(key_col)[numeric_cols].mean()

    # 행이 너무 많으면 첫 번째 measure 기준 상위 max_rows만 사용
    if len(sub) > max_rows:
        sub = sub.nlargest(max_rows, numeric_cols[0])

    normalized = _normalize_with_direction(sub, numeric_cols)

    top3 = {col: normalized[col].nlargest(3).round(4).to_dict() for col in numeric_cols}

    fig_h = max(min(len(sub) * 0.45 + 2, 22), 6)
    fig_w = max(len(numeric_cols) * 2.2 + 2, 8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        normalized, annot=True, fmt=".2f", cmap="YlOrRd",
        ax=ax, linewidths=0.4, linecolor="#eeeeee",
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.6},
    )
    ax.set_title("Category × Metric Heatmap (Normalized)", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelsize=9, rotation=30)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "heatmap_matrix.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return {"chart_path": path, "stats": {"top3_per_metric": top3}}


def plot_bubble(df: pd.DataFrame, key_col: str = None, measure_cols: list = None) -> dict:
    """
    카테고리별 멀티 지표 버블 차트.
    x=measure[0], y=measure[1], 크기=measure[2], 색=measure[3]
    4개 지표를 한 장에 표현.
    """
    cat_cols     = list(df.select_dtypes(include=["object"]).columns)
    numeric_cols = _get_numeric_cols(df, measure_cols)

    if len(numeric_cols) < 2:
        return {"chart_path": None, "stats": {}}
    if key_col is None or key_col not in df.columns:
        key_col = cat_cols[0] if cat_cols else None
    if key_col is None:
        return {"chart_path": None, "stats": {}}

    x_col     = numeric_cols[0]
    y_col     = numeric_cols[1]
    size_col  = numeric_cols[2] if len(numeric_cols) > 2 else None
    color_col = numeric_cols[3] if len(numeric_cols) > 3 else None

    cols_needed = [key_col, x_col, y_col] + ([size_col] if size_col else []) + ([color_col] if color_col else [])
    sub = df[cols_needed].dropna()

    # 버블 크기 정규화 (50~800)
    if size_col:
        sv = sub[size_col]
        sizes = ((sv - sv.min()) / (sv.max() - sv.min() + 1e-9) * 750 + 50).values
    else:
        sizes = 120

    # 색상
    if color_col:
        cv = sub[color_col]
        c_norm = (cv - cv.min()) / (cv.max() - cv.min() + 1e-9)
        sc_kwargs = dict(c=c_norm, cmap="RdYlGn")
    else:
        sc_kwargs = dict(color=PALETTE_MAIN)

    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(sub[x_col], sub[y_col], s=sizes, alpha=0.7,
                    edgecolors="white", linewidth=0.8, **sc_kwargs)

    if color_col:
        cbar = plt.colorbar(sc, ax=ax, shrink=0.65)
        cbar.set_label(color_col, fontsize=9)

    # 평균선 (사분면 기준)
    ax.axvline(sub[x_col].mean(), color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(sub[y_col].mean(), color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # 주목할 포인트 라벨링 (y 상위 4 + 하위 3)
    notable = pd.concat([sub.nlargest(4, y_col), sub.nsmallest(3, y_col)]).drop_duplicates()
    for _, row in notable.iterrows():
        ax.annotate(str(row[key_col])[:18], (row[x_col], row[y_col]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points", color="#333333")

    # 음수 없는 지표면 축을 0부터 시작
    if sub[x_col].min() >= 0:
        ax.set_xlim(left=0)
    if sub[y_col].min() >= 0:
        ax.set_ylim(bottom=0)

    size_label  = f"  |크기: {size_col}"  if size_col  else ""
    color_label = f"  |색: {color_col}" if color_col else ""
    _apply_style(ax, f"Bubble: {x_col} vs {y_col}{size_label}{color_label}",
                 xlabel=x_col, ylabel=y_col)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"bubble_{x_col}_vs_{y_col}.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)

    # 사분면 분류 (평균 기준) — count만 반환
    x_mean = float(sub[x_col].mean())
    y_mean = float(sub[y_col].mean())
    sub["_qx"] = sub[x_col].apply(lambda v: "high_x" if v >= x_mean else "low_x")
    sub["_qy"] = sub[y_col].apply(lambda v: "high_y" if v >= y_mean else "low_y")
    quadrants = sub.groupby(["_qx", "_qy"]).size().to_dict()
    quadrants = {f"{k[0]}_{k[1]}": int(v) for k, v in quadrants.items()}
    sub = sub.drop(columns=["_qx", "_qy"])

    stats = {
        "axes":         {"x": x_col, "y": y_col, "size": size_col, "color": color_col},
        "x_mean":       round(x_mean, 4),
        "y_mean":       round(y_mean, 4),
        "quadrants":    quadrants,
        "top5_by_y":    sub.nlargest(5, y_col)[[key_col, y_col]].set_index(key_col)[y_col].round(4).to_dict(),
        "bottom5_by_y": sub.nsmallest(5, y_col)[[key_col, y_col]].set_index(key_col)[y_col].round(4).to_dict(),
    }
    return {"chart_path": path, "stats": stats}


def plot_radar(df: pd.DataFrame, key_col: str = None, measure_cols: list = None, top_n: int = 8) -> dict:
    """
    상위 N개 카테고리의 레이더(스파이더) 차트.
    정규화된 지표를 다각형으로 표현 — 카테고리별 강점/약점 한눈에 비교.
    """
    cat_cols     = list(df.select_dtypes(include=["object"]).columns)
    numeric_cols = _get_numeric_cols(df, measure_cols)

    if len(numeric_cols) < 3:
        return {"chart_path": None, "stats": {}}
    if key_col is None or key_col not in df.columns:
        key_col = cat_cols[0] if cat_cols else None
    if key_col is None:
        return {"chart_path": None, "stats": {}}

    sub = df[[key_col] + numeric_cols].dropna().groupby(key_col)[numeric_cols].mean()

    # 전체 기준 정규화 후 상위 N 카테고리 선택 (방향 보정 포함)
    normalized = _normalize_with_direction(sub, numeric_cols)
    top_cats   = sub[numeric_cols[0]].nlargest(top_n).index
    normalized = normalized.loc[top_cats]

    N      = len(numeric_cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = sns.color_palette("tab10", len(normalized))

    for (cat, row), color in zip(normalized.iterrows(), colors):
        values = row.tolist() + [row.iloc[0]]
        ax.plot(angles, values, linewidth=1.8, color=color, label=str(cat)[:22])
        ax.fill(angles, values, alpha=0.07, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(numeric_cols, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="gray")
    ax.grid(True, linewidth=0.5, alpha=0.5)
    inverted = [c for c in numeric_cols if _is_lower_better(c)]
    inv_note = f"  ↓better: {', '.join(inverted)}" if inverted else ""
    ax.set_title(f"Radar: Top {top_n} Categories (Normalized{inv_note})", fontsize=11, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8, frameon=False)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "radar_top_categories.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)

    # 종합 점수 (정규화 지표 평균) 및 차별화 지표
    composite = normalized.mean(axis=1).round(4)
    metric_std = normalized.std(axis=0).round(4)  # 높을수록 카테고리 간 차이 큰 지표

    stats = {
        "composite_scores":            composite.to_dict(),
        "top3_composite":              composite.nlargest(3).to_dict(),
        "most_differentiating_metric": str(metric_std.idxmax()),   # 카테고리 간 격차 가장 큰 지표
        "least_differentiating_metric": str(metric_std.idxmin()),  # 카테고리 간 격차 가장 작은 지표
        "metric_variance":             metric_std.to_dict(),
    }
    return {"chart_path": path, "stats": stats}


def plot_grouped_bar(df: pd.DataFrame, key_col: str = None, measure_cols: list = None, top_n: int = 12) -> dict:
    """
    카테고리 × 지표 그룹 바차트 (정규화).
    지표별로 색이 다른 막대를 나란히 배치 — 카테고리 간 종합 성과 비교.
    """
    cat_cols     = list(df.select_dtypes(include=["object"]).columns)
    numeric_cols = _get_numeric_cols(df, measure_cols)

    if not numeric_cols:
        return {"chart_path": None, "stats": {}}
    if key_col is None or key_col not in df.columns:
        key_col = cat_cols[0] if cat_cols else None
    if key_col is None:
        return {"chart_path": None, "stats": {}}

    sub = df[[key_col] + numeric_cols].dropna().groupby(key_col)[numeric_cols].mean()

    # 전체 기준 정규화 후 첫 번째 지표 기준 상위 top_n 선택 (방향 보정 포함)
    normalized = _normalize_with_direction(sub, numeric_cols)
    top_cats   = sub[numeric_cols[0]].nlargest(top_n).index
    plot_df    = normalized.loc[top_cats]

    x      = np.arange(len(plot_df))
    n_cols = len(numeric_cols)
    width  = 0.75 / n_cols
    colors = sns.color_palette("tab10", n_cols)

    fig, ax = plt.subplots(figsize=(max(13, len(plot_df) * 0.9), 5))
    for i, (col, color) in enumerate(zip(numeric_cols, colors)):
        offset = (i - n_cols / 2 + 0.5) * width
        ax.bar(x + offset, plot_df[col], width * 0.92,
               label=col, color=color, alpha=0.82, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([str(c)[:16] for c in plot_df.index], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Normalized Score (0–1)", fontsize=9)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    inverted = [c for c in numeric_cols if _is_lower_better(c)]
    inv_note = f"  ↓better: {', '.join(inverted)}" if inverted else ""
    _apply_style(ax, f"Grouped Bar: Top {top_n} by {numeric_cols[0]} (Normalized{inv_note})")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "grouped_bar_top_categories.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)

    # 종합 점수 및 균형 점수
    composite = plot_df.mean(axis=1).round(4)
    balance   = (1 - plot_df.std(axis=1)).round(4)  # 1에 가까울수록 지표 간 고른 성과

    stats = {
        "composite_scores": composite.to_dict(),       # 정규화 지표 평균 (높을수록 전반적 우수)
        "balance_scores":   balance.to_dict(),          # 지표 간 균형 (높을수록 한 지표에 편중 안 됨)
        "top3_composite":   composite.nlargest(3).to_dict(),
        "metric_leaders":   {col: str(plot_df[col].idxmax()) for col in numeric_cols},  # 지표별 1위 카테고리
    }
    return {"chart_path": path, "stats": stats}


# ─────────────────────────────
# Relationship
# ─────────────────────────────

def plot_correlation(df: pd.DataFrame, measure_cols: list = None) -> dict:
    """수치형 컬럼 간 상관관계 히트맵 + 상관계수 행렬"""
    cols = _get_numeric_cols(df, measure_cols)
    numeric_df = df[cols] if cols else df.select_dtypes(include=["float64", "int64"])
    if numeric_df.shape[1] < 2:
        return {"chart_path": None, "stats": {}}

    corr_matrix = numeric_df.corr().round(3)

    strong_pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= 0.3:
                strong_pairs.append({
                    "pair": f"{cols[i]} vs {cols[j]}",
                    "correlation": round(float(val), 3),
                    "direction": "양의 상관" if val > 0 else "음의 상관"
                })

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(max(len(cols) * 1.2 + 2, 7), max(len(cols) * 1.0 + 1, 6)))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        ax=ax, linewidths=0.5, linecolor="#eeeeee",
        vmin=-1, vmax=1,
        annot_kws={"size": 9},
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Correlation Heatmap", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelsize=9, rotation=30)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return {
        "chart_path": path,
        "stats": {
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_pairs": strong_pairs,
        }
    }


def plot_scatter_pairs(df: pd.DataFrame, top_n_pairs: int = 5, measure_cols: list = None) -> dict:
    """수치형 컬럼 쌍별 scatter plot + 피어슨 상관계수 (상관 절댓값 상위 N쌍만)"""
    paths = []
    stats = {}
    numeric_cols = _get_numeric_cols(df, measure_cols)
    if len(numeric_cols) < 2:
        return {"chart_paths": [], "stats": {}}

    # 전체 쌍의 상관계수 계산 후 절댓값 상위 top_n_pairs만 추출
    all_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            x_col, y_col = numeric_cols[i], numeric_cols[j]
            pair_df = df[[x_col, y_col]].dropna()
            if len(pair_df) < 2:
                continue
            r = pair_df[x_col].corr(pair_df[y_col])
            all_pairs.append((abs(r), x_col, y_col, round(float(r), 3)))

    all_pairs.sort(key=lambda x: x[0], reverse=True)
    selected_pairs = all_pairs[:top_n_pairs]

    for _, x_col, y_col, corr_val in selected_pairs:
        pair_df = df[[x_col, y_col]].dropna()
        stats[f"{x_col} vs {y_col}"] = {"pearson_r": corr_val}

        color = PALETTE_POS if corr_val >= 0 else PALETTE_NEG
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(pair_df[x_col], pair_df[y_col],
                   alpha=0.45, s=20, color=color, edgecolors="none")
        try:
            z = np.polyfit(pair_df[x_col], pair_df[y_col], 1)
            p = np.poly1d(z)
            xs = np.linspace(pair_df[x_col].min(), pair_df[x_col].max(), 200)
            ax.plot(xs, p(xs), color="black", linewidth=1.2, linestyle="--", alpha=0.7)
        except Exception:
            pass
        ax.set_ylim(pair_df[y_col].min() - pair_df[y_col].std() * 0.3,
                    pair_df[y_col].max() + pair_df[y_col].std() * 0.3)
        _apply_style(ax, f"Scatter: {x_col} vs {y_col}  (r={corr_val})", xlabel=x_col, ylabel=y_col)
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"scatter_{x_col}_vs_{y_col}.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        paths.append(path)
    return {"chart_paths": paths, "stats": stats}


# ─────────────────────────────
# Time
# ─────────────────────────────

def plot_timeseries(df: pd.DataFrame, measure_cols: list = None, time_cols: list = None) -> dict:
    """datetime 컬럼 기준 수치형 지표 시계열 추세 + 기간/범위"""
    paths = []
    stats = {}
    # LLM이 분류한 time_cols 우선, 없으면 dtype/이름 휴리스틱 폴백
    if not time_cols:
        time_cols = [c for c in df.columns if "datetime" in str(df[c].dtype) or "date" in c.lower()]
    numeric_cols = _get_numeric_cols(df, measure_cols)

    if not time_cols or len(numeric_cols) == 0:
        return {"chart_paths": [], "stats": {}}

    time_col  = time_cols[0]
    df_sorted = df.sort_values(time_col)

    for metric in numeric_cols:
        s = df_sorted[metric].dropna()
        stats[metric] = {
            "start": str(df_sorted[time_col].min()),
            "end":   str(df_sorted[time_col].max()),
            "first_value": round(float(df_sorted[metric].iloc[0]), 4),
            "last_value":  round(float(df_sorted[metric].iloc[-1]), 4),
            "overall_change_pct": round(
                (float(df_sorted[metric].iloc[-1]) - float(df_sorted[metric].iloc[0]))
                / (abs(float(df_sorted[metric].iloc[0])) + 1e-9) * 100, 2
            ),
        }
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.fill_between(df_sorted[time_col], df_sorted[metric],
                        alpha=0.15, color=PALETTE_MAIN)
        ax.plot(df_sorted[time_col], df_sorted[metric],
                color=PALETTE_MAIN, linewidth=1.6, marker="o", markersize=3)
        _apply_style(ax, f"Timeseries: {metric}", xlabel=time_col, ylabel=metric)
        plt.xticks(rotation=40)
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"ts_{metric}.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        paths.append(path)
    return {"chart_paths": paths, "stats": stats}


def plot_seasonality(df: pd.DataFrame, measure_cols: list = None, time_cols: list = None) -> dict:
    """월/요일 기준 시즌성 bar chart + 피크 시점"""
    paths = []
    stats = {}
    # LLM이 분류한 time_cols 우선, 없으면 dtype/이름 휴리스틱 폴백
    if not time_cols:
        time_cols = [c for c in df.columns if "datetime" in str(df[c].dtype) or "date" in c.lower()]
    numeric_cols = _get_numeric_cols(df, measure_cols)

    if not time_cols or len(numeric_cols) == 0:
        return {"chart_paths": [], "stats": {}}

    time_col = time_cols[0]
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df["_month"]   = df[time_col].dt.month
    df["_weekday"] = df[time_col].dt.day_name()

    for period_col, label in [("_month", "month"), ("_weekday", "weekday")]:
        for metric in numeric_cols:
            agg = df.groupby(period_col)[metric].mean().round(4)
            stats[f"{label}_{metric}"] = {
                "peak":         str(agg.idxmax()),
                "peak_value":   round(float(agg.max()), 4),
                "trough":       str(agg.idxmin()),
                "trough_value": round(float(agg.min()), 4),
            }
            palette = sns.color_palette(PALETTE_SEQ, len(agg))
            fig, ax = plt.subplots(figsize=(9, 4))
            bars = ax.bar(agg.index.astype(str), agg.values,
                          color=palette, edgecolor="white", linewidth=0.5)
            peak_val = agg.max()
            for bar, val in zip(bars, agg.values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + peak_val * 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
            _apply_style(ax, f"Seasonality ({label}): {metric}", xlabel=label, ylabel=f"avg {metric}")
            plt.xticks(rotation=30 if label == "weekday" else 0, fontsize=9)
            fig.tight_layout()
            path = os.path.join(OUTPUT_DIR, f"season_{label}_{metric}.png")
            fig.savefig(path, bbox_inches="tight", dpi=120)
            plt.close(fig)
            paths.append(path)
    return {"chart_paths": paths, "stats": stats}


# ─────────────────────────────
# 클러스터링 차트
# ─────────────────────────────

def plot_cluster_profile(cluster_centers: pd.DataFrame, k: int) -> dict:
    """클러스터별 지표 평균 프로파일 히트맵 (z-score 정규화)."""
    if cluster_centers.empty:
        return {"chart_paths": []}

    fig, ax = plt.subplots(figsize=(max(7, len(cluster_centers.columns) * 1.4), max(3, k * 0.8 + 1.5)))
    sns.heatmap(
        cluster_centers,
        annot=True, fmt=".2f",
        cmap="RdYlGn", center=0,
        linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Cluster Profile (z-score normalized)", fontsize=12, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cluster_profile.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return {"chart_paths": [path]}


def plot_cluster_scatter(df: pd.DataFrame, x_col: str, y_col: str, cluster_col: str = "cluster", key_col: str = None) -> dict:
    """클러스터별 색상 scatter plot."""
    if x_col not in df.columns or y_col not in df.columns:
        return {"chart_paths": []}

    k = df[cluster_col].nunique()
    palette = sns.color_palette("Set2", k)
    fig, ax = plt.subplots(figsize=(8, 5))
    for cid, color in zip(sorted(df[cluster_col].unique()), palette):
        mask = df[cluster_col] == cid
        ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                   label=f"Cluster {cid}", color=color, alpha=0.75, s=60, edgecolors="white", linewidth=0.5)
        if key_col and key_col in df.columns and df[mask].shape[0] <= 30:
            for _, row in df[mask].iterrows():
                ax.annotate(str(row[key_col])[:10], (row[x_col], row[y_col]),
                            fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    _apply_style(ax, f"Cluster: {x_col} vs {y_col}", xlabel=x_col, ylabel=y_col)
    ax.legend(title="Cluster", fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"cluster_scatter_{x_col}_vs_{y_col}.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return {"chart_paths": [path]}


# ─────────────────────────────
# 그룹별 분포 (범주 × 수치, 다변수 교차)
# ─────────────────────────────

def plot_grouped_box(df: pd.DataFrame, key_col: str = None, measure_cols: list = None,
                     top_n: int = 10, min_rows_per_group: int = 5) -> dict:
    """카테고리별 수치형 '분포'를 그룹 박스플롯으로 비교.

    평균 막대가 숨기는 것(분산·이상치)을 드러낸다. 그룹당 행이 여러 개인 원본(raw)
    데이터에서만 의미가 있으므로, 그룹당 1행(집계본)이면 그릴 게 없어 스킵한다.
    """
    cat_cols = list(df.select_dtypes(include=["object"]).columns)
    numeric_cols = _get_numeric_cols(df, measure_cols)
    if key_col is None or key_col not in df.columns:
        key_col = cat_cols[0] if cat_cols else None
    if key_col is None or not numeric_cols:
        return {"chart_paths": [], "stats": {}}

    # feasibility: 그룹당 행이 사실상 1개면(집계본) 분포 비교 불가
    sizes = df.groupby(key_col).size()
    if float(sizes.median()) < 2:
        return {"chart_paths": [], "stats": {},
                "skipped": f"{key_col} 그룹당 행이 1개뿐(집계본) — 분포 비교 불가"}

    # 표본 충분한 그룹 중 행수 상위 top_n
    valid = sizes[sizes >= min_rows_per_group]
    if valid.empty:
        return {"chart_paths": [], "stats": {},
                "skipped": f"그룹별 표본이 모두 {min_rows_per_group} 미만 — 분포 비교 부적합"}
    top_cats = valid.nlargest(top_n).index.tolist()

    paths, stats = [], {}
    for metric in numeric_cols:
        groups, labels, per = [], [], {}
        for cat in top_cats:
            s = df.loc[df[key_col] == cat, metric].dropna()
            if len(s) < min_rows_per_group:
                continue
            q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
            groups.append(s.values)
            labels.append(str(cat)[:18])
            per[str(cat)] = {"median": round(float(s.median()), 2), "q1": round(q1, 2),
                             "q3": round(q3, 2), "iqr": round(q3 - q1, 2), "n": int(len(s))}
        if not groups:
            continue

        colors = sns.color_palette(PALETTE_SEQ, len(groups))[::-1]
        fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.85), 5))
        bp = ax.boxplot(groups, vert=True, patch_artist=True, showfliers=True,
                        medianprops=dict(color=PALETTE_ACCENT, linewidth=2),
                        flierprops=dict(marker="o", markersize=3, alpha=0.4, markerfacecolor=PALETTE_NEG,
                                        markeredgecolor="none"))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("white")
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        _apply_style(ax, f"{metric} distribution by {key_col} (top {len(groups)})", ylabel=metric)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"groupedbox_{metric}.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        paths.append(path)

        widest = max(per.items(), key=lambda kv: kv[1]["iqr"]) if per else None
        stats[metric] = {
            "by_group": per,
            "widest_spread_group": widest[0] if widest else None,
            "note": (f"{widest[0]} 그룹의 산포(IQR={widest[1]['iqr']})가 가장 큼 — 평균만으론 안 보이는 편차"
                     if widest else ""),
        }
    return {"chart_paths": paths, "stats": stats, "key_col": key_col}


def plot_distribution_by_target(df: pd.DataFrame, target_col: str = None, measure_cols: list = None,
                                max_levels: int = 8, n_bins: int = 4, min_rows: int = 5) -> dict:
    """target(결과변수) 수준별로 다른 수치의 분포를 비교 — 'X가 target에 영향을 주는가'에 정조준.

    target이 이산(예: 리뷰점수 1~5)이면 그 값으로, 연속이면 사분위 구간으로 묶는다.
    그룹당 표본이 부족하면(집계본 등) 스킵.
    """
    numeric_cols = _get_numeric_cols(df, measure_cols)
    if (not target_col or target_col not in df.columns
            or not pd.api.types.is_numeric_dtype(df[target_col])):
        return {"chart_paths": [], "stats": {}, "target": target_col,
                "skipped": "target 수치 컬럼 없음"}

    t = df[target_col]
    rounded = t.round()
    nun = int(rounded.dropna().nunique())   # 평균값 등 소수점 섞여도 정수 기준 수준 수로 판단
    if nun <= 1:
        return {"chart_paths": [], "stats": {}, "target": target_col, "skipped": "target 값 단일"}

    if nun <= max_levels:
        buckets = rounded.astype("Int64")
        levels = sorted(b for b in buckets.dropna().unique())
        label = {b: str(int(b)) for b in levels}
    else:
        try:
            buckets = pd.qcut(t, n_bins, duplicates="drop")
        except Exception:  # noqa: BLE001
            return {"chart_paths": [], "stats": {}, "target": target_col, "skipped": "구간화 실패"}
        levels = list(buckets.cat.categories)
        label = {lv: f"Q{i + 1}" for i, lv in enumerate(levels)}

    work = df.assign(_bucket=buckets)
    paths, stats = [], {}
    for metric in numeric_cols:
        if metric == target_col:
            continue
        groups, labels, per = [], [], {}
        for lv in levels:
            s = work.loc[work["_bucket"] == lv, metric].dropna()
            if len(s) < min_rows:
                continue
            groups.append(s.values)
            labels.append(label[lv])
            per[label[lv]] = {"median": round(float(s.median()), 2), "n": int(len(s))}
        if len(groups) < 2:
            continue

        colors = sns.color_palette("RdYlGn", len(groups))
        fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.1), 5))
        bp = ax.boxplot(groups, patch_artist=True, showfliers=True,
                        medianprops=dict(color="black", linewidth=2),
                        flierprops=dict(marker="o", markersize=3, alpha=0.3,
                                        markerfacecolor=PALETTE_NEG, markeredgecolor="none"))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor("white")
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        _apply_style(ax, f"{metric} distribution by {target_col}", xlabel=target_col, ylabel=metric)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"distbytarget_{metric}.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        paths.append(path)

        meds = [v["median"] for v in per.values()]
        trend = ("target 낮을수록 높음" if meds[0] > meds[-1]
                 else "target 낮을수록 낮음" if meds[0] < meds[-1] else "수준 간 차이 작음")
        stats[metric] = {"target": target_col, "by_level": per, "trend": trend}
    return {"chart_paths": paths, "stats": stats, "target": target_col}


def plot_multiline_timeseries(df: pd.DataFrame, time_col: str = None, key_col: str = None,
                              measure_cols: list = None, top_n: int = 6, min_periods: int = 3) -> dict:
    """카테고리별 시간 추세를 한 그래프에 여러 줄로(상위 N개) — 시간 × 범주 교차."""
    numeric_cols = _get_numeric_cols(df, measure_cols)
    cat_cols = list(df.select_dtypes(include=["object"]).columns)
    if not time_col:
        time_col = next((c for c in df.columns
                         if "datetime" in str(df[c].dtype) or "date" in c.lower() or "month" in c.lower()), None)
    if key_col is None or key_col not in df.columns:
        key_col = cat_cols[0] if cat_cols else None
    if not time_col or not key_col or not numeric_cols:
        return {"chart_paths": [], "stats": {}, "skipped": "시간/범주/수치 컬럼 부족"}

    d = df.copy()
    period = pd.to_datetime(d[time_col], errors="coerce").dt.to_period("M").astype(str)
    if period.notna().sum() == 0:
        return {"chart_paths": [], "stats": {}, "skipped": f"{time_col} 시간 파싱 불가"}
    d["_period"] = period

    paths, stats = [], {}
    for metric in numeric_cols:
        agg = d.groupby(["_period", key_col])[metric].mean().reset_index()
        top_cats = d.groupby(key_col)[metric].sum().nlargest(top_n).index
        wide = (agg[agg[key_col].isin(top_cats)]
                .pivot(index="_period", columns=key_col, values=metric).sort_index())
        if len(wide) < min_periods or wide.shape[1] == 0:
            continue

        colors = sns.color_palette("tab10", wide.shape[1])
        fig, ax = plt.subplots(figsize=(max(9, len(wide) * 0.5), 5))
        for (col, series), color in zip(wide.items(), colors):
            ax.plot(range(len(wide)), series.values, marker="o", markersize=3,
                    linewidth=1.6, color=color, label=str(col)[:18])
        ax.set_xticks(range(len(wide)))
        ax.set_xticklabels(list(wide.index), rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
        _apply_style(ax, f"{metric} trend by {key_col} (top {wide.shape[1]})", ylabel=metric)
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"multiline_{metric}.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        paths.append(path)

        per = {}
        for col in wide.columns:
            s = wide[col].dropna()
            if len(s) >= 2:
                chg = (float(s.iloc[-1]) - float(s.iloc[0])) / (abs(float(s.iloc[0])) + 1e-9) * 100
                per[str(col)] = {"start": round(float(s.iloc[0]), 2), "end": round(float(s.iloc[-1]), 2),
                                 "change_pct": round(chg, 1), "peak_period": str(s.idxmax())}
        stats[metric] = {"by_category": per, "periods": int(len(wide))}
    return {"chart_paths": paths, "stats": stats, "key_col": key_col, "time_col": time_col}


def plot_crosstab_heatmap(df: pd.DataFrame, cat_a: str = None, cat_b: str = None,
                          max_card: int = 15, max_overall_card: int = 50) -> dict:
    """두 범주형 변수의 교차 빈도 히트맵 — 범주 × 범주."""
    all_cats = list(df.select_dtypes(include=["object"]).columns)
    # cat_a가 명시되면 카디널리티 무관하게 사용(상위 N개만 표시하므로). 미지정 시 저카디널리티 우선.
    if cat_a is None:
        low = [c for c in all_cats if df[c].nunique() <= max_overall_card]
        if len(low) < 2:
            return {"chart_paths": [], "stats": {}, "skipped": "범주형 컬럼 2개 미만 — 교차 불가"}
        cat_a, cat_b = low[0], low[1]
    elif cat_b is None:
        others = [c for c in all_cats if c != cat_a]
        if not others:
            return {"chart_paths": [], "stats": {}, "skipped": "교차할 두 번째 범주형 컬럼 없음"}
        cat_b = min(others, key=lambda c: df[c].nunique())  # 가장 저카디널리티 짝
    if cat_a not in df.columns or cat_b not in df.columns:
        return {"chart_paths": [], "stats": {}, "skipped": "지정 범주 컬럼 없음"}

    top_a = df[cat_a].value_counts().nlargest(max_card).index
    top_b = df[cat_b].value_counts().nlargest(max_card).index
    sub = df[df[cat_a].isin(top_a) & df[cat_b].isin(top_b)]
    ct = pd.crosstab(sub[cat_a], sub[cat_b])
    if ct.size == 0:
        return {"chart_paths": [], "stats": {}, "skipped": "교차표 비어있음"}

    fig_w = max(8, ct.shape[1] * 0.7 + 2)
    fig_h = max(5, ct.shape[0] * 0.5 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax,
                linewidths=0.4, linecolor="#eeeeee", cbar_kws={"shrink": 0.6}, annot_kws={"size": 7})
    ax.set_title(f"Crosstab: {cat_a} x {cat_b} (counts)", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="x", labelsize=8, rotation=40)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"crosstab_{cat_a}_x_{cat_b}.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)

    total = int(ct.values.sum())
    flat = ct.stack()
    top_cell = flat.idxmax()
    stats = {
        "cat_a": cat_a, "cat_b": cat_b, "total": total,
        "top_combo": {"a": str(top_cell[0]), "b": str(top_cell[1]), "count": int(flat.max())},
        "shape": list(ct.shape),
    }
    return {"chart_paths": [path], "stats": stats, "cat_a": cat_a, "cat_b": cat_b}

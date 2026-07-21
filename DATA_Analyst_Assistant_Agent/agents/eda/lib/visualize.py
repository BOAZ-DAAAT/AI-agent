import os
import glob
import json
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from DATA_Analyst_Assistant_Agent.agents.eda.lib.dtype_utils import categorical_object_columns, usable_time_columns

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


def clear_output_dirs():
    """이전 런이 outputs/all·key에 남긴 PNG를 지운다 — 런마다 무한 누적되는 것 방지.

    app.invoke() 시작 전, 런당 1회만 호출한다(모듈 import 시나 plot 함수 내부에서 호출하면
    다른 런/테스트가 같은 프로세스에서 동시에 그린 파일까지 지울 수 있어 위험하다).
    """
    for d in (OUTPUT_DIR, KEY_DIR):
        for f in glob.glob(os.path.join(d, "*.png")):
            os.remove(f)

# ─────────────────────────────
# 공통 스타일 설정
# ─────────────────────────────

PALETTE_MAIN   = "#4C72B0"
PALETTE_ACCENT = "#DD8452"
PALETTE_NEG    = "#C44E52"
PALETTE_POS    = "#55A868"
PALETTE_SEQ    = "Blues"
PALETTE_SOFT   = "#8DA0CB"
GRID_COLOR     = "#D8DEE9"
TEXT_MUTED     = "#5B6574"
FACE_COLOR     = "#F8FAFC"

sns.set_theme(
    style="whitegrid",
    context="notebook",
    rc={
        "axes.facecolor": "#FFFFFF",
        "figure.facecolor": FACE_COLOR,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.7,
        "axes.edgecolor": "#D5DBE3",
        "axes.labelcolor": "#1F2937",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "font.family": ["DejaVu Sans", "sans-serif"],
    },
)


# ─────────────────────────────
# semantic 가드 (#71 A) — E2E 실측 잡차트 방지:
#   ID·일련번호(값 크기에 의미 없음)는 모든 차트에서, 0/1 플래그는 분포·산점류에서 제외.
#   플래그는 '그룹별 비율'(집계 차트의 mean)로만 의미가 있다.
# ─────────────────────────────
_ID_NAME_RE = re.compile(r"(^|_)(id|uuid|guid|seq|sequential|idx|index)($|_)", re.IGNORECASE)
# r_quartile/f_quartile/m_quartile처럼 연속값을 몇 구간으로 나눠 만든 순서형 파생 컬럼 —
# dtype은 수치형이라 통과되지만 산점도로 그리면 몇 개 수평 줄로 눌려 보여 정보량이 낮고,
# 대개 원본 컬럼(예: monetary_value)에서 파생돼 사실상 중복 정보다(#166 실측 피드백).
_ORDINAL_BUCKET_NAME_RE = re.compile(
    r"(_quartile|_quantile|_decile|_percentile|_tier|_bucket|_bin|_grade|_rank)$", re.IGNORECASE
)
_KEY_MAX_CARDINALITY = 50            # 이보다 범주가 많으면 차트 라벨 축으로 부적합(ID급)
_MAX_PLOT_POINTS = 8000              # raw 포인트 차트(violin·scatter·box) 렌더 샘플 상한 — 96k행 렌더 폭증 방지


def _plot_sample(data, cap: int = _MAX_PLOT_POINTS):
    """그리기 전용 샘플 — 통계는 전체로 계산하고 렌더만 줄인다(시각 차이 무시 가능, 재현 고정)."""
    if len(data) <= cap:
        return data
    return data.sample(cap, random_state=42)


def _is_binary_flag(s: pd.Series) -> bool:
    """0/1(불리언) 플래그 — 히스토그램·박스·산점도 축은 무의미(두 줄짜리 그림)."""
    if pd.api.types.is_bool_dtype(s):
        return True
    u = pd.unique(s.dropna())
    try:
        return 0 < len(u) <= 2 and set(float(v) for v in u).issubset({0.0, 1.0})
    except (TypeError, ValueError):
        return False


def _is_id_like_numeric(col: str, s: pd.Series) -> bool:
    """수치형 ID·일련번호 — 이름 신호 또는 거의 전부 유니크한 정수."""
    if _ID_NAME_RE.search(col):
        return True
    n = len(s)
    return n > 0 and pd.api.types.is_integer_dtype(s) and s.nunique() > 0.9 * n


def _valid_cat_col(df: pd.DataFrame, col, max_card: int = _KEY_MAX_CARDINALITY):
    """차트 라벨 축으로 쓸 범주 컬럼 검증 — 없거나 고카디널리티(고객ID 96k 등)거나
    상수(스냅샷 anchor_date처럼 그룹이 1개뿐)면 None."""
    if not col or col not in df.columns:
        return None
    n = df[col].nunique(dropna=True)
    if n > max_card or n < 2:
        return None
    return col


def _pick_key_col(df: pd.DataFrame, key_col=None, max_card: int = _KEY_MAX_CARDINALITY):
    """key_col이 유효하면 그대로, 아니면 저카디널리티 범주 컬럼으로 폴백. 없으면 None(차트 스킵)."""
    valid = _valid_cat_col(df, key_col, max_card)
    if valid:
        return valid
    for c in categorical_object_columns(df):
        if _valid_cat_col(df, c, max_card):
            return c
    return None


# 집계 마트(고객 단위 RFM 등)는 범주형 컬럼이 없어 _pick_key_col이 None을 반환하기 쉽다.
# 이 경우 0/1 세그먼트 플래그를 그룹 키로 쓴다 — 우선순위: 복합 세그먼트 > 단일 세그먼트 > 기타 flag.
_FLAG_KEY_PRIORITY = ("is_high_value_low_satisfaction", "is_high_value", "is_low_satisfaction")


def _pick_flag_key_col(df: pd.DataFrame):
    """범주형 그룹 키가 없을 때 쓸 0/1 플래그 컬럼을 우선순위대로 고른다. 없으면 None."""
    for col in _FLAG_KEY_PRIORITY:
        if col in df.columns and _is_binary_flag(df[col].dropna()) and df[col].nunique(dropna=True) >= 2:
            return col
    for col in df.columns:
        if str(col).lower().startswith("is_") and _is_binary_flag(df[col].dropna()) and df[col].nunique(dropna=True) >= 2:
            return col
    return None


def _get_numeric_cols(df: pd.DataFrame, measure_cols: list = None, allow_flags: bool = True) -> list:
    """measure_cols가 있으면 그 중 실제 수치형만, 없으면 전체 수치형 컬럼 반환.
    semantic 가드: ID류는 항상 제외, allow_flags=False면 0/1 플래그도 제외(분포·산점류용)."""
    if measure_cols:
        cols = [c for c in measure_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    else:
        cols = list(df.select_dtypes(include=["float64", "int64"]).columns)
    out = []
    for c in cols:
        s = df[c].dropna()
        if s.empty or _is_id_like_numeric(c, s):
            continue
        if not allow_flags and _is_binary_flag(s):
            continue
        out.append(c)
    return out


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

def _pretty_label(label: str) -> str:
    return str(label).replace("_", " ").strip()


def _ellipsize(value, max_len: int = 18) -> str:
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _flag_group_label(flag_col: str, value) -> str:
    """0/1 플래그를 그룹 키로 쓸 때, 값 대신 사람이 읽을 라벨을 만든다.

    is_high_value_low_satisfaction 같은 flag는 그룹핑 후 값이 그냥 0/1로 남아 축 라벨이
    "0", "1"로만 보인다(#166 실측 피드백 — 어떤 그룹 비교인지 안 읽힘). 컬럼명에서 뜻을
    뽑아 1=해당 세그먼트, 0=Rest로 표기한다.
    """
    name = _pretty_label(re.sub(r"^is_", "", str(flag_col)))
    try:
        is_positive = float(value) == 1.0
    except (TypeError, ValueError):
        return str(value)
    return name.capitalize() if is_positive else "Rest"


_BADGE_POSITIONS = {
    "upper right": (0.99, 0.98, "right", "top"),
    "upper left":  (0.01, 0.98, "left",  "top"),
    "lower right": (0.99, 0.02, "right", "bottom"),
    "lower left":  (0.01, 0.02, "left",  "bottom"),
}


def _add_stat_badge(ax, lines: list[str], loc: str = "upper right"):
    if not lines:
        return
    x, y, ha, va = _BADGE_POSITIONS.get(loc, _BADGE_POSITIONS["upper right"])
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8,
        color=TEXT_MUTED,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#FFFFFF",
            "edgecolor": "#D9E2EC",
            "alpha": 0.96,
        },
    )


def _apply_style(ax, title, xlabel="", ylabel="", subtitle: str = "", grid_axis: str = "y"):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=18, loc="left")
    if subtitle:
        ax.text(0.0, 1.005, subtitle, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9, color=TEXT_MUTED)
    if xlabel:
        ax.set_xlabel(_pretty_label(xlabel), fontsize=10)
    if ylabel:
        ax.set_ylabel(_pretty_label(ylabel), fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D5DBE3")
    ax.spines["bottom"].set_color("#D5DBE3")
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.7, alpha=0.5)
    if grid_axis == "y":
        ax.grid(axis="x", visible=False)
    elif grid_axis == "x":
        ax.grid(axis="y", visible=False)


def _format_category_ticks(ax, rotation: int = 30, max_len: int = 18, axis: str = "x"):
    if axis == "x":
        labels = [_ellipsize(t.get_text(), max_len) for t in ax.get_xticklabels()]
        ax.set_xticklabels(labels, rotation=rotation, ha="right", fontsize=8.5)
    else:
        labels = [_ellipsize(t.get_text(), max_len) for t in ax.get_yticklabels()]
        ax.set_yticklabels(labels, fontsize=8.5)


# ─────────────────────────────
# Distribution
# ─────────────────────────────

def plot_distributions(df: pd.DataFrame, measure_cols: list = None) -> dict:
    """수치형 컬럼 히스토그램 + 분포 통계"""
    paths = []
    stats = {}
    for col in _get_numeric_cols(df, measure_cols, allow_flags=False):
        s = df[col].dropna()
        stats[col] = {
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "skewness": round(float(s.skew()), 4),
        }
        # 왜도 큰 분포는 선형축에서 막대 하나로 뭉개진다 — 자기 통계의 log_transform 처방을 차트가 소비.
        # 0을 포함할 수 있어 log1p 변환을 쓴다(순수 log는 0원 고객 몇 명에 무력화되는 실측 있음).
        log_x = stats[col]["skewness"] > 2 and float(s.min()) >= 0
        plot_s = np.log1p(s) if log_x else s
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(plot_s, bins=28, kde=True, stat="count",
                     color=PALETTE_MAIN, edgecolor="white", linewidth=0.6, alpha=0.72, ax=ax)
        mean_v = np.log1p(float(s.mean())) if log_x else float(s.mean())
        med_v = np.log1p(float(s.median())) if log_x else float(s.median())
        ax.axvline(mean_v, color=PALETTE_ACCENT, linestyle="--", linewidth=1.4, label=f"mean={stats[col]['mean']}")
        ax.axvline(med_v,  color=PALETTE_NEG,    linestyle=":",  linewidth=1.4, label=f"median={stats[col]['median']}")
        ax.legend(fontsize=8, frameon=False)
        _apply_style(
            ax,
            f"Distribution: {_pretty_label(col)}" + (" (log1p)" if log_x else ""),
            xlabel=(f"log1p({col})" if log_x else col),
            ylabel="Count",
            subtitle="Histogram with density overlay for shape and tail inspection",
        )
        _add_stat_badge(ax, [
            f"median {stats[col]['median']:.2f}",
            f"IQR {(float(s.quantile(0.75)) - float(s.quantile(0.25))):.2f}",
            f"skew {stats[col]['skewness']:.2f}",
        ])
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"dist_{col}.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        paths.append(path)
    return {"chart_paths": paths, "stats": stats}


def plot_ecdfs(df: pd.DataFrame, measure_cols: list = None) -> dict:
    """수치형 컬럼 ECDF. 분위수와 꼬리 분포를 빠르게 보여준다."""
    paths = []
    stats = {}
    for col in _get_numeric_cols(df, measure_cols, allow_flags=False):
        s = df[col].dropna()
        if len(s) < 5:
            continue
        log_x = float(s.skew()) > 2 and float(s.min()) >= 0
        plot_s = np.log1p(s) if log_x else s
        q50 = float(np.log1p(s.quantile(0.5))) if log_x else float(s.quantile(0.5))
        q90 = float(np.log1p(s.quantile(0.9))) if log_x else float(s.quantile(0.9))
        q99 = float(np.log1p(s.quantile(0.99))) if log_x else float(s.quantile(0.99))
        fig, ax = plt.subplots(figsize=(7, 4.2))
        sns.ecdfplot(plot_s, ax=ax, color=PALETTE_MAIN, linewidth=2.2)
        ax.axvline(q50, color=PALETTE_ACCENT, linestyle="--", linewidth=1.2)
        ax.axvline(q90, color=PALETTE_POS, linestyle=":", linewidth=1.2)
        ax.axvline(q99, color=PALETTE_NEG, linestyle=":", linewidth=1.2)
        _apply_style(
            ax,
            f"ECDF: {_pretty_label(col)}" + (" (log1p)" if log_x else ""),
            xlabel=(f"log1p({col})" if log_x else col),
            ylabel="Cumulative share",
            subtitle="Percentile curve to expose concentration and long-tail behavior",
        )
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        _add_stat_badge(ax, [
            f"p50 {float(s.quantile(0.5)):.2f}",
            f"p90 {float(s.quantile(0.9)):.2f}",
            f"p99 {float(s.quantile(0.99)):.2f}",
        ])
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"ecdf_{col}.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        paths.append(path)
        stats[col] = {
            "p50": round(float(s.quantile(0.5)), 4),
            "p90": round(float(s.quantile(0.9)), 4),
            "p99": round(float(s.quantile(0.99)), 4),
        }
    return {"chart_paths": paths, "stats": stats}


def plot_boxplots(df: pd.DataFrame, measure_cols: list = None) -> dict:
    """수치형 컬럼 박스플롯 + IQR 통계"""
    paths = []
    stats = {}
    for col in _get_numeric_cols(df, measure_cols, allow_flags=False):
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
        log_y = float(s.skew()) > 2 and float(s.min()) >= 0  # 왜도 처방 소비 — 이상치에 짓눌린 박스 방지
        fig, ax = plt.subplots(figsize=(5, 5))
        s_plot = _plot_sample(np.log1p(s) if log_y else s)
        bp = ax.boxplot(
            s_plot, orientation="vertical", patch_artist=True,
            boxprops=dict(facecolor=PALETTE_MAIN, alpha=0.6, linewidth=1.2),
            medianprops=dict(color=PALETTE_ACCENT, linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", color=PALETTE_NEG, alpha=0.5, markersize=4),
        )
        ax.set_xticks([])
        _apply_style(
            ax,
            f"Boxplot: {_pretty_label(col)}" + (" (log1p)" if log_y else ""),
            ylabel=(f"log1p({col})" if log_y else col),
            subtitle="Median, spread and outlier fences at a glance",
        )
        ax.grid(axis="x", visible=False)
        _add_stat_badge(ax, [
            f"Q1 {stats[col]['q1']:.2f}",
            f"median {stats[col]['median']:.2f}",
            f"Q3 {stats[col]['q3']:.2f}",
        ])
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
    numeric_cols = _get_numeric_cols(df, measure_cols, allow_flags=False)
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
        log_y = stats[col]["skewness"] > 2 and float(s.min()) >= 0  # 왜도 처방 소비
        fig, ax = plt.subplots(figsize=(5, 6))
        parts = ax.violinplot(_plot_sample(np.log1p(s) if log_y else s).values,
                              orientation="vertical", showmedians=True, showextrema=True)
        parts["cmedians"].set_color(PALETTE_ACCENT)
        parts["cmedians"].set_linewidth(2)
        for pc in parts["bodies"]:
            pc.set_facecolor(PALETTE_MAIN)
            pc.set_alpha(0.6)
            pc.set_edgecolor("white")
        ax.set_xticks([])
        _apply_style(
            ax,
            f"Violin: {_pretty_label(col)}" + (" (log1p)" if log_y else ""),
            ylabel=(f"log1p({col})" if log_y else col),
            subtitle="Shape-aware distribution view with density width",
        )
        _add_stat_badge(ax, [
            f"median {stats[col]['median']:.2f}",
            f"IQR {stats[col]['iqr']:.2f}",
            f"std {stats[col]['std']:.2f}",
        ])
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
    for col in categorical_object_columns(df):
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
        _apply_style(
            ax,
            f"Category Distribution: {_pretty_label(col)}",
            ylabel="Count",
            subtitle="Top categories ranked by observed frequency",
        )
        _format_category_ticks(ax, rotation=35, max_len=16)
        _add_stat_badge(ax, [f"categories {int(df[col].nunique())}", f"top1 share {float(vc.iloc[0] / max(len(df[col].dropna()), 1)):.1%}"])
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"catdist_{col}.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        paths.append(path)
    return {"chart_paths": paths, "stats": stats}


# ─────────────────────────────
# Comparison
# ─────────────────────────────

def plot_top_n_barplot(df: pd.DataFrame, top_n: int = 10, key_col: str = None, measure_cols: list = None,
                       top_only: bool = False) -> dict:
    """카테고리 키 기준 수치형 지표 상위/하위 N개 barplot + 실제 값.
    top_only=True면 상위만 그린다(codegen처럼 이미 정렬·상위추출된 결과에 하위 잉여 방지)."""
    paths = []
    stats = {}
    numeric_cols = _get_numeric_cols(df, measure_cols)   # 플래그 허용 — 그룹 mean = 비율이라 유의미
    key_col = _pick_key_col(df, key_col)                 # 고카디널리티(고객ID 등) 라벨축 차단
    if key_col is None or len(numeric_cols) == 0:
        return {"chart_paths": [], "stats": {}}
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
            if label == "bottom" and top_only:
                continue                                       # 상위만 요청됨(codegen 등)
            # bottom이 전부 0이면 스킵
            if label == "bottom" and subset[metric].max() == 0:
                continue
            fig, ax = plt.subplots(figsize=(9, 5))
            vals  = subset[metric].values
            names = [_ellipsize(v, 18) for v in subset[key_col].values]
            max_val = max(vals.max(), 1e-9)  # 0 나눔 방지
            bars  = ax.barh(range(len(names)), vals, color=color, alpha=0.82, edgecolor="white")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlim(left=0)
            for bar, val in zip(bars, vals):
                label_val = abs(val) if abs(val) > 1e-9 else 0.0  # -0.000 방지
                ax.text(bar.get_width() + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{label_val:.3f}", va="center", fontsize=8)
            _apply_style(
                ax,
                f"{label.upper()} {top_n}: {_pretty_label(metric)}",
                xlabel=metric,
                subtitle="Group mean ranking for the selected metric",
                grid_axis="x",
            )
            ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
            ax.grid(axis="y", visible=False)
            ax.spines["left"].set_visible(False)
            _add_stat_badge(ax, [f"groups {len(subset)}", f"range {float(vals.min()):.2f} - {float(vals.max()):.2f}"])
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
    numeric_cols = _get_numeric_cols(df, measure_cols)
    key_col = _pick_key_col(df, key_col)                 # 고카디널리티 라벨축 차단
    if key_col is None or not numeric_cols:
        return {"chart_path": None, "stats": {}}
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
    numeric_cols = _get_numeric_cols(df, measure_cols, allow_flags=False)  # 플래그 축 버블 방지
    if len(numeric_cols) < 2:
        return {"chart_path": None, "stats": {}}
    key_col = _pick_key_col(df, key_col)
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

    # 한글은 title.family=DejaVu Sans(한글 글리프 없음)에서 빈 박스로 깨진다 — 영어로 고정.
    size_label  = f"  |size: {size_col}"  if size_col  else ""
    color_label = f"  |color: {color_col}" if color_col else ""
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
    numeric_cols = _get_numeric_cols(df, measure_cols)

    if len(numeric_cols) < 3:
        return {"chart_path": None, "stats": {}}
    key_col = _pick_key_col(df, key_col)
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
    numeric_cols = _get_numeric_cols(df, measure_cols)

    if not numeric_cols:
        return {"chart_path": None, "stats": {}}
    key_col = _pick_key_col(df, key_col)
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
    ax.set_xticklabels([_ellipsize(c, 16) for c in plot_df.index], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Normalized Score (0–1)", fontsize=9)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    inverted = [c for c in numeric_cols if _is_lower_better(c)]
    inv_note = f"  ↓better: {', '.join(inverted)}" if inverted else ""
    _apply_style(
        ax,
        f"Grouped Bar: Top {top_n} by {_pretty_label(numeric_cols[0])} (Normalized{inv_note})",
        subtitle="Multi-metric category comparison on a unified 0-1 scale",
    )
    _add_stat_badge(ax, [f"groups {len(plot_df)}", f"metrics {len(numeric_cols)}"])
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

def plot_mean_ci_comparison(df: pd.DataFrame, key_col: str = None, measure_cols: list = None,
                            top_n: int = 8) -> dict:
    """그룹 평균과 95% CI를 함께 보여주는 비교 차트."""
    numeric_cols = _get_numeric_cols(df, measure_cols, allow_flags=False)
    key_col = _pick_key_col(df, key_col) or _pick_flag_key_col(df)
    if key_col is None or not numeric_cols:
        return {"chart_paths": [], "stats": {}}

    # 0/1 플래그를 그룹 키로 쓴 경우 축 라벨이 "0","1"로만 보여 어떤 비교인지 안 읽힌다
    # (#166 실측 피드백) — 컬럼명 기반 라벨(예: "High value low satisfaction" / "Rest")로 대체.
    is_flag_group = str(key_col).lower().startswith("is_") and _is_binary_flag(df[key_col].dropna())

    paths = []
    stats = {}
    for metric in numeric_cols[:2]:
        sub = df[[key_col, metric]].dropna()
        grouped = sub.groupby(key_col)[metric].agg(["mean", "std", "count"]).reset_index()
        grouped = grouped[grouped["count"] >= 2].sort_values("mean", ascending=False).head(top_n)
        if len(grouped) < 2:
            continue
        grouped["ci95"] = 1.96 * (grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1)))
        fig, ax = plt.subplots(figsize=(9, max(4.8, len(grouped) * 0.55)))
        ypos = np.arange(len(grouped))
        ax.errorbar(
            grouped["mean"],
            ypos,
            xerr=grouped["ci95"],
            fmt="o",
            color=PALETTE_MAIN,
            ecolor=PALETTE_SOFT,
            elinewidth=2,
            capsize=3,
            markersize=7,
        )
        ax.set_yticks(ypos)
        if is_flag_group:
            tick_labels = [_flag_group_label(key_col, v) for v in grouped[key_col]]
        else:
            tick_labels = [_ellipsize(v, 18) for v in grouped[key_col]]
        ax.set_yticklabels(tick_labels, fontsize=9)
        ax.invert_yaxis()
        _apply_style(
            ax,
            f"Mean +/- 95% CI: {_pretty_label(metric)} by {_pretty_label(key_col)}",
            xlabel=metric,
            subtitle="Group averages with uncertainty bands",
            grid_axis="x",
        )
        # 평균 내림차순 정렬이라 최고 평균 그룹은 항상 (y=0=상단, x=최대값=우측)에 찍힌다 —
        # 기본 위치(우상단)에 배지를 두면 그 점을 매번 가린다(#166 실측 발견). 안전한
        # 좌상단(항상 최고 평균보다 왼쪽)으로 옮긴다.
        _add_stat_badge(ax, [f"groups {len(grouped)}", f"top mean {float(grouped['mean'].iloc[0]):.2f}"],
                        loc="upper left")
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"interval_{metric}.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        paths.append(path)
        stats[metric] = {
            "top_groups": grouped[[key_col, "mean", "ci95", "count"]].round(4).to_dict(orient="records")
        }
    return {"chart_paths": paths, "stats": stats}


def plot_segment_flag_profiles(df: pd.DataFrame, measure_cols: list = None, max_flags: int = 2) -> dict:
    """0/1 세그먼트 플래그 기준으로 주요 수치 지표의 표준화 차이를 비교한다."""
    flag_cols = []
    for col in df.columns:
        if str(col).lower().startswith("is_") and _is_binary_flag(df[col]):
            flag_cols.append(col)
    numeric_cols = _get_numeric_cols(df, measure_cols, allow_flags=False)
    if not flag_cols or not numeric_cols:
        return {"chart_paths": [], "stats": {}}

    paths = []
    stats = {}
    metric_pool = numeric_cols[:6]
    for flag in flag_cols[:max_flags]:
        work = df[[flag] + metric_pool].dropna()
        if work.empty or work[flag].nunique(dropna=True) < 2:
            continue
        flag_values = work[flag].astype(float)
        seg = work[flag_values == 1.0]
        base = work[flag_values == 0.0]
        if len(seg) < 10 or len(base) < 10:
            continue
        rows = []
        for metric in metric_pool:
            overall_std = float(work[metric].std())
            if overall_std <= 1e-9:
                continue
            effect = (float(seg[metric].mean()) - float(base[metric].mean())) / overall_std
            rows.append({
                "metric": metric,
                "effect": effect,
                "seg_mean": float(seg[metric].mean()),
                "base_mean": float(base[metric].mean()),
            })
        if len(rows) < 2:
            continue
        prof = pd.DataFrame(rows).sort_values("effect")
        fig, ax = plt.subplots(figsize=(9, max(4.5, len(prof) * 0.6)))
        colors = [PALETTE_POS if v >= 0 else PALETTE_NEG for v in prof["effect"]]
        ax.barh(range(len(prof)), prof["effect"], color=colors, alpha=0.85, edgecolor="white")
        ax.axvline(0, color="#64748B", linewidth=1)
        ax.set_yticks(range(len(prof)))
        ax.set_yticklabels([_pretty_label(m) for m in prof["metric"]], fontsize=9)
        for idx, value in enumerate(prof["effect"]):
            x = value + (0.03 if value >= 0 else -0.03)
            ha = "left" if value >= 0 else "right"
            ax.text(x, idx, f"{value:+.2f}std", va="center", ha=ha, fontsize=8, color="#334155")
        _apply_style(
            ax,
            f"Segment Profile: {_pretty_label(flag)}",
            xlabel="Standardized mean delta vs. rest",
            subtitle="Positive values indicate the flagged segment is higher than the rest",
            grid_axis="x",
        )
        _add_stat_badge(ax, [f"segment {float(flag_values.mean()):.1%}", f"n {len(seg):,} vs {len(base):,}"])
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"segment_profile_{flag}.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        paths.append(path)
        stats[flag] = {
            "segment_rate": round(float(flag_values.mean()), 4),
            "effects": prof.round(4).to_dict(orient="records"),
        }
    return {"chart_paths": paths, "stats": stats}


def plot_correlation(df: pd.DataFrame, measure_cols: list = None) -> dict:
    """수치형 컬럼 간 상관관계 히트맵 + 상관계수 행렬"""
    cols = _get_numeric_cols(df, measure_cols, allow_flags=False)   # 플래그 상관은 노이즈
    numeric_df = df[cols] if cols else df.select_dtypes(include=["float64", "int64"])
    # 상수 컬럼은 상관이 NaN → 히트맵에 빈 행/열로 노출되므로 제거
    numeric_df = numeric_df.loc[:, numeric_df.std(numeric_only=True) > 0]
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
    _apply_style(
        ax,
        "Correlation Heatmap",
        subtitle="Pairwise linear relationships among selected numeric metrics",
        grid_axis="x",
    )
    ax.tick_params(axis="x", labelsize=9, rotation=30)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    _format_category_ticks(ax, rotation=30, max_len=18, axis="x")
    _format_category_ticks(ax, max_len=18, axis="y")
    _add_stat_badge(ax, [f"metrics {len(cols)}", f"strong pairs {len(strong_pairs)}"])
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
    numeric_cols = _get_numeric_cols(df, measure_cols, allow_flags=False)   # 0/1 플래그 산점도 방지
    numeric_cols = [c for c in numeric_cols if not _ORDINAL_BUCKET_NAME_RE.search(c)]  # quartile류 제외
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
            if abs(r) >= 0.98:
                continue  # 동어반복 — 정의상 종속(파생 컬럼)인 쌍은 '관계 발견'이 아니다
            all_pairs.append((abs(r), x_col, y_col, round(float(r), 3)))

    all_pairs.sort(key=lambda x: x[0], reverse=True)
    selected_pairs = all_pairs[:top_n_pairs]

    for _, x_col, y_col, corr_val in selected_pairs:
        pair_df = df[[x_col, y_col]].dropna()
        stats[f"{x_col} vs {y_col}"] = {"pearson_r": corr_val}

        color = PALETTE_POS if corr_val >= 0 else PALETTE_NEG
        draw_df = _plot_sample(pair_df)                 # 렌더 샘플 — 상관은 전체로 계산됨
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(draw_df[x_col], draw_df[y_col],
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
        _apply_style(ax, f"Scatter: {x_col} vs {y_col}", xlabel=x_col, ylabel=y_col)
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
    # LLM이 분류한 time_cols 우선, 없으면 dtype/이름 휴리스틱 폴백 → 둘 다 usable 게이트 통과해야 함
    candidates = time_cols or [c for c in df.columns if "datetime" in str(df[c].dtype) or "date" in c.lower()]
    time_cols = usable_time_columns(df, candidates)
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
    # LLM이 분류한 time_cols 우선, 없으면 dtype/이름 휴리스틱 폴백 → 둘 다 usable 게이트 통과해야 함
    candidates = time_cols or [c for c in df.columns if "datetime" in str(df[c].dtype) or "date" in c.lower()]
    time_cols = usable_time_columns(df, candidates)
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
    numeric_cols = _get_numeric_cols(df, measure_cols)
    key_col = _pick_key_col(df, key_col)
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
            groups.append(_plot_sample(s).values)        # 렌더 샘플 (통계 per는 전체 기준)
            labels.append(str(cat)[:18])
            per[str(cat)] = {"median": round(float(s.median()), 2), "q1": round(q1, 2),
                             "q3": round(q3, 2), "iqr": round(q3 - q1, 2), "n": int(len(s))}
        if not groups:
            continue

        colors = sns.color_palette(PALETTE_SEQ, len(groups))[::-1]
        fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.85), 5))
        bp = ax.boxplot(groups, orientation="vertical", patch_artist=True, showfliers=True,
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
            groups.append(_plot_sample(s).values)        # 렌더 샘플 (통계는 전체 기준)
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
    if not time_col:
        candidates = [c for c in df.columns
                     if "datetime" in str(df[c].dtype) or "date" in c.lower() or "month" in c.lower()]
        usable = usable_time_columns(df, candidates)
        time_col = usable[0] if usable else None
    elif time_col not in usable_time_columns(df, [time_col]):
        time_col = None
    # 다른 key_col 소비 함수(plot_top_n_barplot·plot_mean_ci_comparison 등)와 달리 이 함수만
    # cat_cols[0]을 검증 없이 쓰고 있었다 — customer_unique_id 같은 ID급 컬럼(9만+ 유니크)이
    # ctx.key_col로 이미 들어와 있으면 그대로 통과되어 "고객 1명당 선 1개, 점 1개"짜리 무의미한
    # 차트가 나왔다(#166 실측). _pick_key_col의 카디널리티/상수 가드를 거치도록 통일한다.
    key_col = _pick_key_col(df, key_col) or _pick_flag_key_col(df)
    # key_col 자신이 measure_cols에 섞여 들어오면(플래그를 key로 쓰는 케이스 등)
    # groupby 결과에 같은 이름 컬럼을 또 넣으려다 충돌한다 — 그룹 키는 지표 후보에서 제외.
    numeric_cols = [c for c in numeric_cols if c != key_col]
    if not time_col or not key_col or not numeric_cols:
        return {"chart_paths": [], "stats": {}, "skipped": "시간/범주/수치 컬럼 부족"}

    d = df.copy()
    period = pd.to_datetime(d[time_col], errors="coerce").dt.to_period("M").astype(str)
    if period.notna().sum() == 0:
        return {"chart_paths": [], "stats": {}, "skipped": f"{time_col} 시간 파싱 불가"}
    d["_period"] = period

    # 0/1 플래그를 그룹 키로 쓴 경우 범례가 "0","1"로만 보여 어떤 그룹인지 안 읽힌다
    # (#166 실측 피드백) — 컬럼명 기반 라벨로 대체.
    is_flag_group = str(key_col).lower().startswith("is_") and _is_binary_flag(df[key_col].dropna())

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
            label = _flag_group_label(key_col, col) if is_flag_group else str(col)[:18]
            ax.plot(range(len(wide)), series.values, marker="o", markersize=3,
                    linewidth=1.6, color=color, label=label)
        ax.set_xticks(range(len(wide)))
        ax.set_xticklabels(list(wide.index), rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
        _apply_style(ax, f"{_pretty_label(metric)} trend by {_pretty_label(key_col)} (top {wide.shape[1]})", ylabel=metric)
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
    all_cats = categorical_object_columns(df)
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
    # 명시 컬럼이라도 ID급(거의 전부 유니크)이면 교차가 무의미 — top-N이 전부 count=1인 0/1 벽이 된다
    for c in (cat_a, cat_b):
        if df[c].nunique(dropna=True) > 0.9 * max(1, len(df)):
            return {"chart_paths": [], "stats": {}, "skipped": f"ID급 컬럼({c}) — 교차표 무의미"}

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

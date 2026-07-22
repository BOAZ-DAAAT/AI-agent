"""도구 실행기(결정론) — look / compute / chart. LLM은 args 만 제안한다.

compute·chart 의 표현식은 공용 표현식 안전 게이트(validate_expression)를 사용하고,
그 위에 인사이트 범위 제한을 더한다: 이 도구는 '인사이트 문장을 만들기 위한 계산기'지
분석 에이전트 2호가 아니다 (허용: 증감률·차이·비율·top/bottom·정렬·집계·reshape /
금지: 회귀·행렬분해 등 — 프롬프트 + _SCOPE_DENY 이중 차단).
차트 렌더링은 matplotlib 결정론 코드 — LLM은 종류·제목만 주문한다.
"""

from __future__ import annotations

import ast
import os
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")                                  # 헤드리스 렌더 (창 없이 PNG 저장)
import matplotlib.pyplot as plt                        # noqa: E402
import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402

from DATA_Analyst_Assistant_Agent.shared.expression_gate import validate_expression

# 보조계산 범위 밖(회귀·행렬분해 등) — 게이트(보안)를 통과해도 역할 경계에서 거부한다.
_SCOPE_DENY = {"polyfit", "lstsq", "svd", "eig", "eigh", "qr", "cholesky", "corrcoef"}
# 숫자 제조 차단 — compute 결과는 검증 corpus 에 편입되므로, df 에서 유도되지 않은 값을
# 만들 수 있는 생성자류를 막는다 (pd.Series([7777.7]) 가 '증거'가 되는 구멍 방지).
_FABRICATION_DENY = {"Series", "DataFrame", "array", "random"}

_MAX_RESULT_CELLS = 120                                # 관찰/corpus 에 실을 결과 상한
_VALID_KINDS = {"line", "bar", "grouped_bar", "table"}

try:                                                   # 한글 라벨 (Windows) — 없으면 무시
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:  # noqa: BLE001
    pass


# ─────────────────────────────
# 표현식 실행 (게이트 → 범위 → eval)
# ─────────────────────────────
def eval_expression(df: pd.DataFrame | None, expression: str) -> tuple[bool, Any]:
    """단일 pandas 표현식을 안전하게 실행. 반환: (ok, 결과 or 거부/오류 사유)."""
    if df is None:
        return False, "no_table: SQL 결과 테이블이 없어 계산할 수 없다"
    if not expression or not isinstance(expression, str):
        return False, "empty_expression"

    gate = validate_expression(expression, list(df.columns))
    if not gate.ok:
        return False, f"gate_rejected: {gate.reason}"

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        return False, f"gate_rejected: not_single_expression: {e.msg}"
    uses_df = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _SCOPE_DENY:
            return False, (f"scope_rejected: {node.attr} — 보조계산 범위 밖(회귀·행렬분해는 분석 에이전트 몫)")
        if isinstance(node, ast.Attribute) and node.attr in _FABRICATION_DENY:
            return False, (f"scope_rejected: {node.attr} — df 에서 유도되지 않은 값 생성 금지"
                           "(compute 결과는 인용 가능 증거가 되므로 df 계산만 허용)")
        if isinstance(node, ast.Name) and node.id == "df":
            uses_df = True
    if not uses_df:                                    # df 무관 표현식(순수 상수 계산 등)은 증거가 될 수 없다
        return False, "scope_rejected: 표현식이 df 를 사용하지 않는다 — df 기반 계산만 증거로 인정"

    try:
        code = compile(tree, "<insight_compute>", "eval")
        result = eval(code, {"__builtins__": {}}, {"df": df.copy(), "pd": pd, "np": np})  # noqa: S307
    except MemoryError:
        return False, "execution_error: resource_memory_error"
    except Exception as exc:  # noqa: BLE001
        return False, f"execution_error: {type(exc).__name__}: {exc}"
    return True, result


def coerce_result(result: Any) -> Any:
    """실행 결과를 JSON 안전 값으로 (관찰·검증 corpus 용, 상한 캡)."""
    if isinstance(result, pd.DataFrame):
        r = result
        if r.shape[0] * max(1, r.shape[1]) > _MAX_RESULT_CELLS:
            r = r.head(max(1, _MAX_RESULT_CELLS // max(1, r.shape[1])))
        if not isinstance(r.index, pd.RangeIndex):
            r = r.reset_index()
        return r.where(pd.notna(r), None).to_dict(orient="records")
    if isinstance(result, pd.Series):
        s = result.head(_MAX_RESULT_CELLS)
        return {str(k): _py(v) for k, v in s.items()}
    return _py(result)


def _py(v: Any) -> Any:
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


# ─────────────────────────────
# look — 증거 들여다보기 (계산 없음, 발췌만)
# ─────────────────────────────
def run_look(pack, args: dict) -> dict:
    target = str(args.get("target", "table"))
    if target == "table":
        return {"ok": True, "target": target, "excerpt": pack.table_summary or "빈 테이블"}
    if target == "sql":
        return {"ok": True, "target": target, "excerpt": pack.generated_sql or "(생성 SQL 없음)"}
    if target in {"eda", "analysis"}:
        payload = pack.eda if target == "eda" else pack.analysis
        if not payload:                                # 없는 증거를 반복해서 보는 배회 방지
            return {"ok": False, "error": f"{target} 결과가 이번 실행에 없다 — 이 target 은 다시 보지 말고 "
                                          "table 요약과 compute 로 답하라"}
        excerpt = _dig(payload, str(args.get("path", "")))
        return {"ok": True, "target": target, "excerpt": excerpt if excerpt is not None else "(해당 경로 없음)"}
    return {"ok": False, "error": f"unknown_target: {target} (table|eda|analysis|sql 중 하나)"}


def _dig(payload: dict, path: str) -> Any:
    """점 표기 경로로 dict 내부 발췌 (예: 'statistical_metadata.group_comparison')."""
    cur: Any = payload
    for key in [p for p in path.split(".") if p]:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


# ─────────────────────────────
# compute — 보조 계산 (게이트 통과 필수)
# ─────────────────────────────
def _dig(payload: dict, path: str) -> Any:
    """Read a dotted path from dict/list payloads. Empty path returns payload."""
    cur: Any = payload
    if not path:
        return cur
    for key in [p for p in path.split(".") if p]:
        if isinstance(cur, dict):
            if key not in cur:
                return None
            cur = cur[key]
            continue
        if isinstance(cur, list) and key.isdigit():
            index = int(key)
            if index >= len(cur):
                return None
            cur = cur[index]
            continue
        return None
    return cur


def run_look(pack, args: dict) -> dict:
    target = str(args.get("target", "table"))
    if target == "table":
        return {"ok": True, "target": target, "excerpt": pack.table_summary or "empty table"}
    if target == "sql":
        return {"ok": True, "target": target, "excerpt": pack.generated_sql or "(no generated SQL)"}

    payloads = {
        "eda": pack.eda,
        "analysis": pack.analysis,
        "eda_raw": getattr(pack, "eda_raw", {}),
        "analysis_raw": getattr(pack, "analysis_raw", {}),
        "analysis_debug": getattr(pack, "analysis_debug", {}),
    }
    if target in payloads:
        payload = payloads[target]
        if not payload:
            return {
                "ok": False,
                "error": (
                    f"{target} evidence is unavailable. Do not repeat this look target; "
                    "use table summary, compute, or another available evidence target. 없다"
                ),
            }
        excerpt = _dig(payload, str(args.get("path", "")))
        artifact_key = target.replace("_raw", "")
        return {
            "ok": True,
            "target": target,
            "artifact_id": getattr(pack, "raw_artifact_ids", {}).get(artifact_key, ""),
            "excerpt": excerpt if excerpt is not None else "(path not found)",
        }
    return {
        "ok": False,
        "error": "unknown_target: use one of table, eda, analysis, eda_raw, analysis_raw, analysis_debug, sql",
    }


def run_compute(pack, args: dict) -> dict:
    expression = str(args.get("expression", ""))
    ok, result = eval_expression(pack.df, expression)
    if not ok:
        return {"ok": False, "expression": expression, "error": result}
    return {"ok": True, "expression": expression, "result": coerce_result(result)}


# ─────────────────────────────
# chart — 차트 주문 (데이터=표현식, 렌더=결정론 코드)
# ─────────────────────────────
def run_chart(pack, args: dict, out_dir: str) -> dict:
    expression = str(args.get("expression", ""))
    kind = str(args.get("kind", "bar"))
    title = str(args.get("title", "")) or "insight_chart"
    if kind not in _VALID_KINDS:
        return {"ok": False, "error": f"invalid_kind: {kind} (line|bar|grouped_bar|table 중 하나)"}

    ok, result = eval_expression(pack.df, expression)
    if not ok:
        return {"ok": False, "expression": expression, "error": result}

    df_r = _to_frame(result)
    if df_r is None or df_r.empty:
        return {"ok": False, "expression": expression,
                "error": "not_chartable: 결과가 스칼라/빈 값 — 그룹별 값을 담은 표현식으로 다시"}

    # 제목-축 불일치 방지: LLM이 그릴 컬럼(y)·라벨(x)을 명시하면 검증 후 그대로 쓴다
    # (미지정 시 첫 수치컬럼 — region 실측에서 '제목=금액, 축=건수' 사고가 났던 지점).
    y_arg = args.get("y")
    y_cols = [str(c) for c in (y_arg if isinstance(y_arg, list) else [y_arg])] if y_arg else []
    for c in y_cols:
        if c not in df_r.columns:
            return {"ok": False, "expression": expression,
                    "error": f"y_column_not_found: {c} — 결과 컬럼 {list(df_r.columns)} 중에서 골라라"}
    x_col = str(args["x"]) if args.get("x") else None
    if x_col and x_col not in df_r.columns:
        return {"ok": False, "expression": expression,
                "error": f"x_column_not_found: {x_col} — 결과 컬럼 {list(df_r.columns)} 중에서 골라라"}

    os.makedirs(out_dir, exist_ok=True)
    filename = _safe_filename(title, kind)
    path = os.path.join(out_dir, filename)
    try:
        _render(df_r, kind, title, path, x_col=x_col, y_cols=y_cols)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "expression": expression, "error": f"render_error: {type(exc).__name__}: {exc}"}
    return {"ok": True, "expression": expression, "kind": kind, "title": title,
            "filename": filename, "local_path": path, "data_preview": coerce_result(result)}


def _to_frame(result: Any) -> pd.DataFrame | None:
    """차트용 프레임 정규화 — 인덱스(범주/시간 축)를 컬럼으로 보존한다."""
    if isinstance(result, pd.Series):
        df_r = result.to_frame(result.name or "value").reset_index()
    elif isinstance(result, pd.DataFrame):
        df_r = result if isinstance(result.index, pd.RangeIndex) else result.reset_index()
    else:
        return None
    df_r = df_r.copy()
    df_r.columns = [str(c) for c in df_r.columns]
    return df_r


def _safe_filename(title: str, kind: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z가-힣]+", "_", title).strip("_")[:40] or "chart"
    return f"insight_{kind}_{slug}.png"


def _fmt_num(v: Any) -> str:
    """차트 라벨/셀용 숫자 포맷 — 과학표기(2.5e+04) 대신 콤마(25,236)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "" if v is None else str(v)
    if f != f:                                          # NaN
        return ""
    if abs(f) >= 1000:
        return f"{f:,.0f}"
    if abs(f) >= 1:
        return f"{f:,.2f}".rstrip("0").rstrip(".")
    return f"{f:.4g}"


def _render(df_r: pd.DataFrame, kind: str, title: str, path: str,
            x_col: str | None = None, y_cols: list[str] | None = None) -> None:
    """결정론 렌더러. x=라벨 축(미지정 시 첫 비수치 컬럼), y=값 컬럼(미지정 시 수치 컬럼)."""
    numeric_cols = [c for c in df_r.columns if pd.api.types.is_numeric_dtype(df_r[c])]
    value_cols = y_cols or numeric_cols                 # LLM 명시가 우선 (제목-축 일치 책임)
    label_col = x_col or next((c for c in df_r.columns if c not in numeric_cols), None)

    if kind == "table":
        show = df_r.head(12)
        fig, ax = plt.subplots(figsize=(min(12, 2 + 1.6 * len(show.columns)), 1 + 0.4 * len(show)))
        ax.axis("off")
        cells = [[(_fmt_num(v) if isinstance(v, (int, float)) else ("" if v is None else str(v)))
                  for v in row] for row in show.itertuples(index=False)]
        table = ax.table(cellText=cells, colLabels=list(show.columns), loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)
    elif kind == "line":
        if not value_cols:
            raise ValueError("line 차트에 수치 컬럼이 없다")
        fig, ax = plt.subplots(figsize=(9, 5))
        x = df_r[label_col].astype(str) if label_col else df_r.index
        for col in value_cols[:4]:
            ax.plot(x, df_r[col], marker="o", markersize=3, label=col)
        ax.legend(fontsize=8)
        if len(df_r) > 8:
            plt.xticks(rotation=45, ha="right", fontsize=8)
    elif kind == "grouped_bar":                         # 다지표 비교 — '특성' 질문용
        if not value_cols:
            raise ValueError("grouped_bar 차트에 수치 컬럼이 없다")
        show = df_r.head(10)
        labels = show[label_col].astype(str) if label_col else show.index.astype(str)
        cols = value_cols[:4]
        xpos = np.arange(len(show))
        width = 0.8 / len(cols)
        fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(show)), 5))
        for i, col in enumerate(cols):
            ax.bar(xpos + i * width, show[col], width, label=col, alpha=0.85, edgecolor="white")
        ax.set_xticks(xpos + width * (len(cols) - 1) / 2)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=8)
    else:  # bar — 값 내림차순 가로 막대 (상위 강조)
        if not value_cols:
            raise ValueError("bar 차트에 수치 컬럼이 없다")
        vcol = value_cols[0]
        show = df_r.nlargest(min(15, len(df_r)), vcol)[::-1]
        labels = show[label_col].astype(str) if label_col else show.index.astype(str)
        fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(show))))
        ax.barh(labels, show[vcol], color="#4C72B0", alpha=0.85, edgecolor="white")
        vmax = max(abs(float(show[vcol].max())), 1e-9)
        for y, v in enumerate(show[vcol]):
            ax.text(float(v) + vmax * 0.01, y, _fmt_num(v), va="center", fontsize=8)
        ax.set_xlabel(vcol, fontsize=9)

    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)

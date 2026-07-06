"""codegen 탈출구 노드 — 6개 도구로 시도조차 못 한 질문에만 발동한다.

route_after_planner(결정론 게이트)가 "실질 분석이 아무 결과도 못 냄"으로 판정할 때만 진입.
흐름:
  ① 판단(gemini/LLM_MODEL): 이 질문이 df 계산 한 줄로 답 가능한 종류인가?
  ② 생성(gpt-5/CODE_GENERATOR_MODEL): CodegenRequest(표현식) 발행
  ③ 게이트(코드): validate_request 통과해야 실행
  ④ 실행: df/pd/np 제한 네임스페이스에서 단일 표현식 eval

성공 → state["codegen"]={status:"success", ...} / 판단no·게이트거부·실행오류 → out_of_domain.
LLM은 제안만 하고, 실행 허가는 게이트가 결정한다(레포 공통 문법).
"""

from __future__ import annotations

import ast
import os
import time
from typing import Any, Dict

import numpy as np
import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import get_context, get_llm, safe_json_parse
from DATA_Analyst_Assistant_Agent.agents.eda.lib.codegen_gate import CodegenRequest, validate_request
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState

_MAX_RESULT_CELLS = 200  # 결과가 이보다 크면 잘라 실음(연산 후 방어 — 하드 리소스 보장은 아님)


def _judge_prompt(question: str, columns: list) -> str:
    return (
        "너는 EDA 보조자다. 아래 질문이 주어진 데이터프레임(df)의 컬럼만으로 "
        "pandas 계산 한 줄(단일 표현식)로 답할 수 있는 종류인지 판단하라.\n"
        "외부 데이터·모델 학습·통계 검정·시각화가 필요하면 불가능(false)이다.\n"
        f"[질문] {question}\n[df 컬럼] {columns}\n"
        '반드시 JSON만 출력: {"computable": true, "reason": "간단한 근거"}'
    )


def _generate_prompt(question: str, columns: list) -> str:
    return (
        "아래 질문에 답하는 pandas 표현식을 하나만 작성하라. 제약:\n"
        "- df(주어진 데이터프레임)·pd·np 만 사용. import·파일 IO·반복문·lambda 금지\n"
        "- 단일 표현식만(여러 줄·할당 금지)\n"
        "- 아래 목록에 있는 컬럼만 사용\n"
        "- 질문이 그룹별 비교·순위·비율이면, 최종 스칼라(예: idxmax) 대신 "
        "그룹별 값을 담은 시리즈/프레임으로 반환하라(시각화에 쓰인다).\n"
        "- 결과가 시각화에 적합하면 chart_hint를 골라라: "
        "bar(카테고리별 값)·line(시간 추세)·grouped_bar(다지표 비교). 부적합하면 null.\n"
        f"[질문] {question}\n[df 컬럼] {columns}\n"
        '반드시 JSON만 출력: {"intent": "무엇을 계산하는지", "target_columns": ["사용 컬럼"], '
        '"expression": "df...", "expected_shape": "scalar|series|frame", '
        '"chart_hint": "bar|line|grouped_bar|null"}'
    )


def _to_frame(result: Any):
    """codegen 결과(series/frame)를 카테고리 축이 컬럼인 df로. 스칼라·기본인덱스 프레임은 None."""
    if isinstance(result, pd.Series):
        df_r = result.to_frame("value").reset_index()
    elif isinstance(result, pd.DataFrame) and not isinstance(result.index, pd.RangeIndex):
        df_r = result.reset_index()
    else:
        return None  # 스칼라·기본인덱스 프레임은 그릴 범주 축이 없음
    df_r.columns = [str(c) for c in df_r.columns]
    return df_r


def _first_value_col(df_r) -> str | None:
    """차트에 쓸 값 컬럼 하나 선택: bool 플래그(is_top 등) 제외한 첫 수치 컬럼.
    → plot 함수가 모든 수치 컬럼을 그려 잡차트 내는 걸 방지.
    (실제 0/1 비율값을 잘못 스킵하지 않도록 dtype이 bool인 것만 제외.)"""
    for col in df_r.columns:
        s = df_r[col]
        if pd.api.types.is_bool_dtype(s):
            continue  # 진짜 플래그(is_top 등)만 스킵
        if pd.api.types.is_numeric_dtype(s):
            return col
    return None


def _render_from_hint(result: Any, chart_hint):
    """LLM이 고른 chart_hint에 따라 기존 visualize 함수를 재사용해 렌더한다.

    판단(어떤 차트)=LLM, 연결=lookup, 렌더=기존 함수, 값컬럼 필터=잡차트 방지.
    OUTPUT_DIR에 PNG를 그리면 chart_selector가 자동으로 주워 key_charts에 넣는다.
    반환: 파일명(provenance 링크) or None. 차트 실패해도 데이터 답은 유지.
    """
    if chart_hint not in {"bar", "line", "grouped_bar"}:
        return None
    df_r = _to_frame(result)
    if df_r is None:
        return None
    try:
        from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize
        if chart_hint == "bar":
            vcol = _first_value_col(df_r)                      # 값 1개만 → 잡차트 방지
            out = visualize.plot_top_n_barplot(df_r, measure_cols=[vcol] if vcol else None)
        elif chart_hint == "grouped_bar":
            out = visualize.plot_grouped_bar(df_r)
        else:  # line — 결과에 시간축 있어야, 없으면 함수가 빈 결과 반환(우아하게)
            out = visualize.plot_multiline_timeseries(df_r)
        paths = out.get("chart_paths", []) if isinstance(out, dict) else []
        return os.path.basename(paths[0]) if paths else None
    except Exception:  # noqa: BLE001
        return None


def _py(v: Any) -> Any:
    """numpy 스칼라 → 파이썬 기본형 (JSON 직렬화용)."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _coerce_result(result: Any):
    """실행 결과를 JSON 안전 값 + 형태 + 잘림 노트로 변환."""
    if isinstance(result, pd.DataFrame):
        note = ""
        r = result
        if result.shape[0] * result.shape[1] > _MAX_RESULT_CELLS:
            r = result.head(max(1, _MAX_RESULT_CELLS // max(1, result.shape[1])))
            note = f"truncated (원래 shape {list(result.shape)})"
        # groupby 등 의미있는 인덱스(지역명 등)는 답 자체라 컬럼으로 보존한다(records가 인덱스를 버림)
        if not isinstance(r.index, pd.RangeIndex):
            r = r.reset_index()
        return r.where(pd.notna(r), None).to_dict(orient="records"), f"frame{list(result.shape)}", note
    if isinstance(result, pd.Series):
        note = ""
        s = result
        if len(result) > _MAX_RESULT_CELLS:
            s = result.head(_MAX_RESULT_CELLS)
            note = f"truncated (원래 len {len(result)})"
        return {str(k): _py(v) for k, v in s.items()}, f"series[{len(result)}]", note
    return _py(result), "scalar", ""


def _out_of_domain(question: str, reason: str) -> Dict[str, Any]:
    return {"codegen": {"status": "out_of_domain", "reason": reason, "user_question": question}}


def codegen_node(state: EDAState) -> dict:
    ctx = get_context()
    df = ctx.df
    q = state.get("user_question", "")

    if df is None or len(df.columns) == 0:
        return _out_of_domain(q, "no_dataframe")
    columns = list(df.columns)

    # ① 판단 (gemini) — df 계산으로 답 가능한 종류인지
    try:
        raw = get_llm().invoke(_judge_prompt(q, columns)).content
        judged = safe_json_parse(raw, {"computable": False, "reason": "판단 파싱 실패"})
    except Exception as exc:  # noqa: BLE001
        return _out_of_domain(q, f"judge_error: {exc}")
    if not judged.get("computable"):
        return _out_of_domain(q, str(judged.get("reason", "not_computable")))

    # ② 생성 (gpt-5) — CodegenRequest 발행
    try:
        raw = get_llm("CODE_GENERATOR_MODEL").invoke(_generate_prompt(q, columns)).content
        req = CodegenRequest.model_validate(safe_json_parse(raw, {}))
    except Exception as exc:  # noqa: BLE001
        return _out_of_domain(q, f"generate_error: {exc}")

    # ③ 게이트 (코드) — 통과해야 실행
    gate = validate_request(req, columns)
    if not gate.ok:
        return _out_of_domain(q, f"gate_rejected: {gate.reason}")

    # ④ 실행 — df/pd/np 만 노출, builtins 제거(게이트가 이미 이름 제한하지만 방어 심층)
    t0 = time.perf_counter()
    try:
        code = compile(ast.parse(req.expression, mode="eval"), "<codegen>", "eval")
        result = eval(code, {"__builtins__": {}}, {"df": df.copy(), "pd": pd, "np": np})  # noqa: S307
    except MemoryError:
        return _out_of_domain(q, "resource_memory_error")
    except Exception as exc:  # noqa: BLE001
        return _out_of_domain(q, f"execution_error: {type(exc).__name__}: {exc}")
    wall_ms = round((time.perf_counter() - t0) * 1000, 1)

    value, out_shape, note = _coerce_result(result)
    chart_file = _render_from_hint(result, req.chart_hint)   # LLM 힌트→기존 함수, key_charts로 흘러감
    return {
        "codegen": {
            "status": "success",
            "intent": req.intent,
            "expression": req.expression,          # provenance — 추적/디버깅용 생성코드 원문
            "expected_shape": req.expected_shape,
            "result": value,
            "result_note": note,
            "chart_hint": req.chart_hint,           # LLM이 고른 차트 종류(provenance)
            "chart": chart_file,                    # 렌더된 차트 파일명(그림은 key_charts에 섞여 나감)
            "telemetry": {                          # v2 승격 판단용 계측
                "input_shape": list(df.shape),
                "output_shape": out_shape,
                "wall_ms": wall_ms,
            },
            "cautions": ["llm_generated"],
        }
    }

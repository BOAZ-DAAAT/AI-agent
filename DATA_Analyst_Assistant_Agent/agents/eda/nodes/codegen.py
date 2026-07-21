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
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict

import numpy as np
import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import (
    accumulate_chart_requests, get_context, get_llm, safe_json_parse,
)
from DATA_Analyst_Assistant_Agent.agents.eda.lib.codegen_gate import CodegenRequest, validate_request
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState
from DATA_Analyst_Assistant_Agent.chart.contract import ChartRequest

_MAX_RESULT_CELLS = 200  # 결과가 이보다 크면 잘라 실음(연산 후 방어 — 하드 리소스 보장은 아님)


def _summarize_llm_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "max_tokens" in text or "more credits" in text or "402" in text:
        return "llm_token_budget_exceeded"
    if "timeout" in text or "timed out" in text:
        return "llm_timeout"
    return f"llm_error:{type(exc).__name__}"


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
        # 보안 게이트가 거부하는 관용구를 미리 알려 처음부터 통과하는 코드를 짜게 유도한다(보안 완화 아님).
        "- 보안 게이트가 거부하니 쓰지 마라: DataFrame.query·DataFrame.eval·pd.eval, "
        "apply/lambda, merge/join/concat, explode, pivot/pivot_table/unstack/get_dummies, "
        "파일 입출력(read_*/to_*/save/load 계열), 대형 배열 생성(np.ones/zeros/arange 등)\n"
        "- 조건 필터링은 query() 문자열이 아니라 반드시 boolean mask와 df.loc[조건식] 형태로 작성하라\n"
        "- 새 지표가 필요하면 boolean/numeric Series 연산으로 만들고 groupby/agg/transform으로 집계하라\n"
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
            # top_only: codegen 결과는 이미 정렬·상위추출됐으니 하위 잉여차트 안 그림
            out = visualize.plot_top_n_barplot(df_r, measure_cols=[vcol] if vcol else None, top_only=True)
        elif chart_hint == "grouped_bar":
            out = visualize.plot_grouped_bar(df_r)
        else:  # line — 결과에 시간축 있어야, 없으면 함수가 빈 결과 반환(우아하게)
            out = visualize.plot_multiline_timeseries(df_r)
        paths = out.get("chart_paths", []) if isinstance(out, dict) else []
        return os.path.basename(paths[0]) if paths else None
    except Exception:  # noqa: BLE001
        return None


def _emit_chart_request(ctx, req: CodegenRequest, value: Any) -> None:
    """codegen 성공 차트를 chart_request 계약으로도 발행한다(PNG 직접 렌더와 별개).

    분석노드 차트처럼 chart_requests.json에 잡혀 report/Phase B가 소비할 수 있게 통일한다.
    stats에 검증된 결과값(환각 방지 닻) + provenance(expression)를 담는다.
    """
    stats: Dict[str, Any] = {"source": "codegen", "expression": req.expression}
    if isinstance(value, dict):                              # series 결과: {범주: 값}
        stats["values"] = {str(k): v for k, v in list(value.items())[:20]}

    df = getattr(ctx, "df", None)

    def _col_type(c: str) -> str:
        # 원본 컬럼은 실제 dtype으로, 계산 중 만든 컬럼은 derived로 표기(unknown 퉁치기 방지).
        if df is None or c not in df.columns:
            return "derived"
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            return "datetime"
        if pd.api.types.is_numeric_dtype(s):
            return "numeric"
        return "categorical"

    try:
        cr = ChartRequest(
            intent=req.intent or "codegen 파생 계산 결과",
            stats=stats,
            columns={c: {"type": _col_type(c)} for c in req.target_columns},
            hint=req.chart_hint,
        ).model_dump()
        accumulate_chart_requests(ctx, [cr])
    except Exception:  # noqa: BLE001 — 주문서 발행 실패해도 데이터 답/PNG는 유지
        pass


def _py(v: Any) -> Any:
    """numpy/pandas 결과를 JSON 직렬화 가능한 기본형으로 변환한다."""
    if v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (pd.Timestamp, pd.Timedelta)):
        return str(v)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return [_py(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _py(val) for k, val in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_py(x) for x in v]
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
        return _py(r.where(pd.notna(r), None).to_dict(orient="records")), f"frame{list(result.shape)}", note
    if isinstance(result, pd.Series):
        note = ""
        s = result
        if len(result) > _MAX_RESULT_CELLS:
            s = result.head(_MAX_RESULT_CELLS)
            note = f"truncated (원래 len {len(result)})"
        return {str(k): _py(v) for k, v in s.items()}, f"series[{len(result)}]", note
    return _py(result), "scalar", ""


def _out_of_domain(question: str, reason: str, *, attempts: int = 0,
                   errors: list | None = None) -> Dict[str, Any]:
    """도메인 밖 판정 시, 그럴듯한 요약을 만들지 않고 정직한 결과를 직접 채운다.

    route_after_codegen이 이걸 보고 insight/hypothesis를 건너뛰고 종료시키므로,
    df 통계로 만든 일반 요약이 out_of_domain과 섞이지 않는다.
    attempts=생성 시도 횟수(judge 실패·no_dataframe이면 0), errors=시도별 실패 사유(계측).
    """
    msg = f"현재 데이터의 컬럼만으로는 이 질문에 답할 수 없습니다. 사유: {reason}"
    caution = {
        "code": "OUT_OF_DOMAIN", "source": "codegen", "severity": "high",
        "message_ko": f"질문이 현재 데이터 컬럼만으로 계산 불가능합니다: {reason}",
        "recommended_action": ["route_to_analysis_or_sql", "clarify_data_scope"],
    }
    codegen: Dict[str, Any] = {
        "status": "out_of_domain", "reason": reason, "user_question": question,
        "attempts": attempts,
    }
    if errors:
        codegen["errors"] = errors                       # attempt별 실패 사유(있을 때만)
    return {
        "codegen": codegen,
        "insight_result": msg,
        "final_summary": msg,
        "hypotheses": "",
        # 정상 흐름(insight)처럼 cautions를 top-level + statistical_metadata 둘 다에 둔다(소비처 애매성 제거)
        "statistical_metadata": {"cautions": [caution]},
        "cautions": [caution],
    }


# 재생성으로 회복 가능한 실패 종류(1회 retry 대상). memory_error는 다시 짜도 위험/무의미 → 제외.
_RETRYABLE_KINDS = {"generate_error", "gate_rejected", "execution_error"}


def _generate_gate_eval(prompt: str, columns: list, df: pd.DataFrame):
    """생성→게이트→실행 한 번. 성공 시 ("ok", req, result, wall_ms),
    실패 시 (kind, reason, prev_expression)를 돌려준다(prev_expression은 재시도 프롬프트용)."""
    try:
        raw = get_llm("CODE_GENERATOR_MODEL").invoke(prompt).content
        req = CodegenRequest.model_validate(safe_json_parse(raw, {}))
    except Exception as exc:  # noqa: BLE001
        return "generate_error", f"generate_error: {exc}", None

    gate = validate_request(req, columns)
    if not gate.ok:
        return "gate_rejected", f"gate_rejected: {gate.reason}", req.expression

    t0 = time.perf_counter()
    try:
        code = compile(ast.parse(req.expression, mode="eval"), "<codegen>", "eval")
        result = eval(code, {"__builtins__": {}}, {"df": df.copy(), "pd": pd, "np": np})  # noqa: S307
    except MemoryError:
        return "memory_error", "resource_memory_error", req.expression
    except Exception as exc:  # noqa: BLE001
        return "execution_error", f"execution_error: {type(exc).__name__}: {exc}", req.expression
    wall_ms = round((time.perf_counter() - t0) * 1000, 1)
    return "ok", req, result, wall_ms


def _retry_prompt(question: str, columns: list, prev_expression: str, error_reason: str) -> str:
    """직전 실패(expression + 사유)를 넣어 다시 짜게 하는 프롬프트. 기존 생성 규칙을 그대로 이어붙인다."""
    return (
        "직전에 작성한 pandas 표현식이 실패했다. 아래 실패 원인을 반영해 다시 작성하라.\n"
        f"[이전 expression] {prev_expression}\n"
        f"[실패 사유] {error_reason}\n"
        "수정 지시:\n"
        "- 날짜/시간 컬럼은 pd.to_datetime(df[\"컬럼\"])로 변환한 뒤 .dt 접근자를 사용하라\n"
        "- query·eval·apply·lambda·merge·join·concat·파일 IO·대형 배열 생성 금지\n"
        "- df·pd·np 만, 단일 표현식만(할당·여러 줄 금지)\n\n"
        + _generate_prompt(question, columns)
    )


def route_after_codegen(state: EDAState):
    """codegen 성공 → insight. 도메인 밖일 때:
      - 실질 도구가 이미 결과를 냄(플래너가 codegen을 오버픽했지만 실패 등) → insight로 정상 분석 살림
      - 아직 시도 안 한 분석 도구가 남아있음 → planner로 복귀(정상 분석 경로에 한 번 더 기회를 준다.
        codegen의 판단 LLM과 planner의 선택 LLM은 서로 다른 잣대라 planner가 codegen으로 풀릴
        거라 오판해도, 아직 안 써본 quality/distribution/comparison 등이 남아있으면 거기서
        답을 찾을 수 있다 — #194, run-019f747b에서 이걸로 EDA 전체가 조기 종료된 사례 확인)
      - 아무 실질 결과도 없고 남은 도구도 없음(진짜 도메인 밖) → 종료(그럴듯한 요약 생성 방지)
    """
    from DATA_Analyst_Assistant_Agent.agents.eda.nodes.planner import (
        _feasible_tools, _substantive_produced_output,
    )
    if (state.get("codegen") or {}).get("status") == "out_of_domain":
        if _substantive_produced_output(state):
            return "insight"
        return "planner" if _feasible_tools(state) else "end"
    return "insight"


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
        return _out_of_domain(q, f"judge_error: {_summarize_llm_error(exc)}")
    if not judged.get("computable"):
        return _out_of_domain(q, str(judged.get("reason", "not_computable")))

    # ②③④ 생성→게이트→실행 (attempt 1). 회복 가능한 실패면 에러 피드백을 넣어 1회만 재시도.
    #   judge는 이미 통과했으므로 다시 돌리지 않는다(생성만 다시 짠다).
    errors: list = []
    outcome = _generate_gate_eval(_generate_prompt(q, columns), columns, df)
    if outcome[0] != "ok":
        kind, reason, prev_expr = outcome
        errors.append(f"attempt1: {reason}")
        if kind not in _RETRYABLE_KINDS:                 # memory_error 등 → 재시도 안 함
            return _out_of_domain(q, reason, attempts=1, errors=errors)
        # attempt 2 — 실패 expression + 사유를 넣어 재생성
        outcome = _generate_gate_eval(_retry_prompt(q, columns, prev_expr or "", reason), columns, df)
        if outcome[0] != "ok":
            reason2 = outcome[1]
            errors.append(f"attempt2: {reason2}")
            return _out_of_domain(q, f"retry_failed: {reason2}", attempts=2, errors=errors)

    _, req, result, wall_ms = outcome
    attempts = 2 if errors else 1                          # errors가 있으면 재시도로 회복된 것
    value, out_shape, note = _coerce_result(result)
    chart_file = _render_from_hint(result, req.chart_hint)   # LLM 힌트→기존 함수, key_charts로 흘러감
    if chart_file and req.chart_hint in {"bar", "line", "grouped_bar"}:
        _emit_chart_request(ctx, req, value)                 # PNG와 별개로 chart_request 계약도 발행
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
                "attempts": attempts,               # 1=첫 시도 성공, 2=재시도로 회복
                "recovered_by_retry": bool(errors), # 재시도가 첫 실패를 건졌는지
            },
            "cautions": ["llm_generated"],
        }
    }

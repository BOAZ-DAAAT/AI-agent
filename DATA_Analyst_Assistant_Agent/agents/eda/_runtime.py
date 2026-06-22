"""EDA 에이전트 공용 런타임.

- LLM 메모이즈 접근자 (get_llm)
- 실행 컨텍스트 홀더 (EdaContext) — 원본 모듈 전역 `_df / _key_col / _measure_cols ...` 대체.
  mini-ReAct 툴과 노드가 동일한 DataFrame/컬럼 정보를 공유하기 위한 것.
  import 시점이 아니라 run() 진입 시 set_context() 로 채워지므로 import 부작용이 없다.
- JSON 파싱 헬퍼

[동작 보존] 원본은 노드마다 `ChatOpenAI(...)` 를 즉석 생성했다. 여기서는
`shared.llm.get_chat_model` 을 1회 메모이즈해 재사용한다(엔진/키 설정은 shared.config 가 정규화).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, List, Optional

from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model
import DATA_Analyst_Assistant_Agent.shared.config  # noqa: F401  (.env 로드 + DB_*/MYSQL_* 별칭 정규화)

MAX_NODE_RETRIES = 2  # 노드당 최대 재시도 횟수


# -----------------------------
# LLM 메모이즈 접근자
# -----------------------------
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = get_chat_model(temperature=0)
    return _llm


# -----------------------------
# 실행 컨텍스트 (원본 모듈 전역 대체)
# -----------------------------
@dataclass
class EdaContext:
    df: Any = None                                  # pandas.DataFrame
    key_col: Optional[str] = None
    measure_cols: Optional[List[str]] = None
    time_cols: Optional[List[str]] = None
    count_col: Optional[str] = None
    target_col: Optional[str] = None                    # 분석 대상(결과변수) — 계약/플래너가 채움
    question_type: str = ""
    priority_metrics: list = field(default_factory=list)
    chart_requests: list = field(default_factory=list)  # skill이 발행한 차트 주문서 누적(중복제거됨)


_context: Optional[EdaContext] = None


def set_context(ctx: EdaContext) -> None:
    global _context
    _context = ctx


def get_context() -> EdaContext:
    global _context
    if _context is None:
        _context = EdaContext()
    return _context


def reset_context() -> None:
    global _context
    _context = None


# -----------------------------
# Utils
# -----------------------------
def safe_json_parse(text_value: str, fallback: dict) -> dict:
    cleaned = (text_value or "").strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return fallback


def _chart_request_sig(req: dict) -> tuple:
    cols = tuple(sorted((req.get("columns") or {}).keys()))
    return (req.get("intent"), cols, req.get("hint"))


def accumulate_chart_requests(ctx: "EdaContext", requests: list) -> None:
    """skill이 발행한 차트 주문서를 ctx에 누적한다(intent·컬럼·hint 기준 중복제거).
    ReAct 재호출/재시도로 같은 주문서가 여러 번 와도 한 번만 쌓인다."""
    if not requests:
        return
    seen = {_chart_request_sig(r) for r in ctx.chart_requests}
    for r in requests:
        sig = _chart_request_sig(r)
        if sig not in seen:
            ctx.chart_requests.append(r)
            seen.add(sig)


def append_errors(state: dict, *errs) -> list:
    """state["error_log"] 에 None이 아닌 에러들을 누적해 새 리스트로 반환."""
    errors = state.get("error_log", [])
    for e in errs:
        if e:
            errors = errors + [e]
    return errors

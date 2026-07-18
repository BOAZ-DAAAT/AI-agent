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
# LLM 메모이즈 접근자 (모델별)
# -----------------------------
_llms: dict[str, Any] = {}


def get_llm(model_env: str = "LLM_MODEL"):
    """model_env별로 메모이즈한 LLM 접근자.

    기본은 LLM_MODEL(공용 텍스트 모델). codegen 등 특수 경로만 전용 모델을
    model_env로 요청한다(예: get_llm("CODE_GENERATOR_MODEL")). 모델별로 따로
    캐시하므로 한 실행에서 서로 다른 모델을 섞어 써도 캐시가 서로를 덮지 않는다.
    """
    if model_env not in _llms:
        _llms[model_env] = get_chat_model(temperature=0, model_env=model_env)
    return _llms[model_env]


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
    user_question: str = ""                              # 질문 프레이밍 신호(예: '저조/최하위') 판단용
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


def split_marked_json(text_value: str, marker: str):
    """LLM 출력에서 marker 뒤에 붙은 JSON을 분리한다.

    서술 프로즈 뒤에 구조화 필드(요약용 facts·우선 가설 등)를 함께 뱉게 하고, 여기서
    프로즈와 JSON을 갈라 각각을 쓰기 위한 헬퍼. marker가 없거나 JSON 파싱이 실패하면
    (원문 그대로, None)을 돌려주므로, 구조화 필드가 없어도 프로즈는 온전히 보존된다.
    반환: (marker 앞 프로즈, 파싱된 JSON 또는 None)
    """
    text = text_value or ""
    idx = text.rfind(marker)
    if idx < 0:
        return text.strip(), None
    prose = text[:idx].strip()
    tail = text[idx + len(marker):].replace("```json", "").replace("```", "").strip()
    try:
        return prose, json.loads(tail)
    except Exception:
        return prose, None


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

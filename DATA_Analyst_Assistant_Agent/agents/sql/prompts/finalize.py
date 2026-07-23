"""Deprecated: 미사용 최종화 프롬프트의 직접 import 호환 모듈."""

from __future__ import annotations

import json
import warnings

from DATA_Analyst_Assistant_Agent.agents.sql._runtime import format_result_rows


def finalize_mart_prompt(state) -> str:
    """Deprecated: 결정론적 ``finalize_answer``를 사용한다."""
    warnings.warn(
        "finalize_mart_prompt는 더 이상 런타임에서 사용되지 않습니다. "
        "결정론적 finalize_answer 경로를 사용하세요.",
        DeprecationWarning,
        stacklevel=2,
    )
    return f"""
너는 데이터 엔지니어/분석가용 결과 요약기다.

사용자 질문:
{state['user_question']}

마트 설계:
{json.dumps(state.get('mart_design', {}), ensure_ascii=False, indent=2)}

생성 SQL:
{state['sql_draft']['sql']}

사전 점검 결과:
{format_result_rows(state.get('precheck_result'))}

사후 점검 결과:
{format_result_rows(state.get('postcheck_result'))}

규칙:
- 한국어
- 마트명, grain, 적재 방식, 핵심 컬럼, 검증 결과를 짧게 요약
- 없는 내용 추측 금지
"""


def finalize_answer_prompt(state) -> str:
    """Deprecated: 결정론적 ``finalize_answer``를 사용한다."""
    warnings.warn(
        "finalize_answer_prompt는 더 이상 런타임에서 사용되지 않습니다. "
        "결정론적 finalize_answer 경로를 사용하세요.",
        DeprecationWarning,
        stacklevel=2,
    )
    return f"""
너는 데이터 분석 답변 작성기다.

사용자 질문:
{state['user_question']}

SQL 결과:
{format_result_rows(state['sql_result'], max_rows=20)}

규칙:
- 한국어
- 핵심 결과 먼저
- 없는 내용 추측 금지
- 결과 범위 안에서만 설명
"""


def finalize_rewrite_prompt(state, answer: str) -> str:
    """Deprecated: 결정론적 ``finalize_answer``를 사용한다."""
    warnings.warn(
        "finalize_rewrite_prompt는 더 이상 런타임에서 사용되지 않습니다. "
        "결정론적 finalize_answer 경로를 사용하세요.",
        DeprecationWarning,
        stacklevel=2,
    )
    return f"""
너는 데이터 분석 결과를 사용자에게 설명하는 한국어 분석가다.

사용자 질문:
{state['user_question']}

SQL 결과:
{format_result_rows(state['sql_result'], max_rows=20)}

초안 답변:
{answer}

다시 작성 규칙:
- 반드시 한국어로 작성한다.
- 마크다운 표를 절대 만들지 않는다.
- 결과 표는 최종 리포트의 SQL Result Table 섹션에서 따로 보여주므로, 여기서는 해석만 쓴다.
- 먼저 1~2문장으로 핵심 결과와 가장 중요한 해석을 설명한다.
- 그 다음 SQL 결과의 각 행을 "- **항목명**: 주요 수치..." 형식의 bullet 목록으로 정리해도 된다.
- bullet 목록에는 배송 지연율, 전체 주문 수, 배송 지연 주문 수, 평균 리뷰 점수를 포함한다.
- 마지막에 1문장으로 리뷰 점수와 배송 지연율 관계 또는 해석 시 주의점을 덧붙인다.
- 숫자는 SQL 결과에 있는 값만 사용한다.
- 마지막 문장은 "따라서"로 시작해 결론을 정리한다.
"""

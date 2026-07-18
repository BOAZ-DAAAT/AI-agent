"""route_after_codegen(#194) 계약 테스트 — 도메인 밖 거부 시 planner 복귀 vs 종료 분기.

run-019f747b 실제 실패 사례: codegen이 '단일 pandas 표현식으로 안 됨'이라 자체 거부했는데
아직 quality/distribution/comparison 등 안 써본 분석 도구가 남아있는데도 EDA 전체가 바로
종료돼버렸다. planner에게 한 번 더 기회를 주도록 고쳤다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import EdaContext, reset_context, set_context
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.codegen import route_after_codegen


@pytest.fixture(autouse=True)
def _context():
    # numeric 2개 + categorical 1개, time 없음 → feasible = quality/distribution/comparison/relationship/clustering
    df = pd.DataFrame({
        "amount": [10, 20, 15, 30, 25, 12, 18, 22, 27, 31, 14, 19, 21, 26, 29],
        "score": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "category": list("ABABABABABABABA"),
    })
    set_context(EdaContext(df=df, question_type="comprehensive"))
    yield
    reset_context()


def _state(**overrides) -> dict:
    base = {"controller_log": [], "codegen": {}}
    base.update(overrides)
    return base


def test_success_goes_to_insight():
    state = _state(codegen={"status": "success"})
    assert route_after_codegen(state) == "insight"


def test_out_of_domain_with_substantive_output_goes_to_insight():
    state = _state(
        codegen={"status": "out_of_domain", "reason": "not_computable"},
        controller_log=[{"round": 0, "choice": "distribution", "reason": "..."},
                        {"round": 1, "choice": "codegen", "reason": "..."}],
        distribution_result="분포 결과 있음",
    )
    assert route_after_codegen(state) == "insight"


def test_out_of_domain_with_feasible_tools_goes_back_to_planner():
    # 아직 시도 안 한 분석 도구(quality/distribution/comparison/relationship/clustering)가 남음
    state = _state(
        codegen={"status": "out_of_domain", "reason": "단일 pandas 표현식으로 수행하기 어렵다"},
        controller_log=[{"round": 0, "choice": "codegen", "reason": "..."}],
    )
    assert route_after_codegen(state) == "planner"


def test_out_of_domain_with_no_feasible_tools_ends():
    # 모든 분석 도구를 이미 다 시도했지만 전부 빈 결과 → 진짜 도메인 밖
    state = _state(
        codegen={"status": "out_of_domain", "reason": "not_computable"},
        controller_log=[
            {"round": 0, "choice": "quality", "reason": "..."},
            {"round": 1, "choice": "distribution", "reason": "..."},
            {"round": 2, "choice": "comparison", "reason": "..."},
            {"round": 3, "choice": "relationship", "reason": "..."},
            {"round": 4, "choice": "clustering", "reason": "..."},
            {"round": 5, "choice": "codegen", "reason": "..."},
        ],
    )
    assert route_after_codegen(state) == "end"

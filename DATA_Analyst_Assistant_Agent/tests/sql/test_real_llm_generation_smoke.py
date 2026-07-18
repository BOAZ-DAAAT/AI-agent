"""실제 SQL 생성 모델의 opt-in smoke 테스트.

실행: RUN_REAL_SQL_LLM=1 python -m pytest \
    DATA_Analyst_Assistant_Agent/tests/sql/test_real_llm_generation_smoke.py -s
"""

from __future__ import annotations

import os

import pytest

from DATA_Analyst_Assistant_Agent.agents.sql.nodes.generate import generate_sql
from DATA_Analyst_Assistant_Agent.tests.sql.test_generation_context import (
    _comprehensive_state,
    _simple_state,
)


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_SQL_LLM") != "1",
    reason="RUN_REAL_SQL_LLM=1일 때만 실제 SQL 생성 모델을 호출합니다",
)


@pytest.mark.parametrize(
    "state",
    [
        pytest.param(_simple_state(), id="simple-query"),
        pytest.param(_comprehensive_state(), id="comprehensive-mart"),
    ],
)
def test_real_model_returns_route_valid_structured_draft(state):
    result = generate_sql(state)

    assert result["generation_failure_reason"] == ""
    assert result["sql_draft"]["sql"]
    expected_type = "create_table_as" if state["plan"]["route_kind"] == "comprehensive" else "select"
    assert result["sql_draft"]["sql_type"] == expected_type

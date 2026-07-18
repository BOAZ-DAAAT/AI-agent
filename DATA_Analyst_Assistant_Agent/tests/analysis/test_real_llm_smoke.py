"""Opt-in real-LLM smoke tests. Run with --run-real-llm.

    pytest <this file> --run-real-llm -s

Split into two live checks that avoid pd.date_range (which segfaults on this
Windows/numpy build, both in test data and in generated code):
  1. classify resolves a daily grain for a one-month question
  2. the full generate->execute->critic loop runs on a non-temporal question

Skipped by default so the normal suite stays offline/deterministic.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest
from dotenv import load_dotenv

load_dotenv()

from DATA_Analyst_Assistant_Agent.agents.analysis.graph import run_analysis_workflow
from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.classify import classify_intent
from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.context import build_analysis_context
from DATA_Analyst_Assistant_Agent.shared.contracts import AnalysisPlan, OrchestrationState

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_LLM") != "1",
    reason="pass --run-real-llm to run live-model smoke test",
)


def _state(query: str, **plan_kwargs) -> OrchestrationState:
    return OrchestrationState(
        run_id="smoke",
        user_query=query,
        goal=query,
        route_kind="comprehensive",
        plan=AnalysisPlan(goal=query, route_kind="comprehensive", **plan_kwargs),
    )


def test_classify_resolves_daily_grain_for_one_month_question() -> None:
    dates = [f"2026-06-{day:02d}" for day in range(1, 31)]
    revenue = [1000 + i * 25 for i in range(len(dates))]
    df = pd.DataFrame({"order_date": dates, "revenue": revenue})
    state = _state("6월 한 달 동안 매출 추이가 어땠는지 분석해줘", metric="revenue")

    context = build_analysis_context(state, df, [])
    intent = classify_intent(context, df)

    print("\n[classify] domain:", intent.domain)
    print("[classify] time_column:", intent.time_column)
    print("[classify] time_grain:", intent.time_grain, "/ span_days:", intent.time_span_days)
    print("[classify] is_time_based:", intent.is_time_based)

    assert intent.time_grain == "D", "one-month span must resolve to daily grain"
    assert intent.time_column == "order_date"
    assert intent.is_time_based is True


def test_full_loop_runs_on_non_temporal_question() -> None:
    df = pd.DataFrame({
        "category": ["A", "A", "B", "B", "C", "C", "A", "B"],
        "revenue": [10.0, 12.0, 25.0, 22.0, 5.0, 6.0, 11.0, 24.0],
    })
    state = _state("카테고리별 매출 비중을 알려줘", metric="revenue", dimension="category")

    result, checks, terminal = run_analysis_workflow(state, df, [])

    intent = result.get("intent") or {}
    crit = result.get("code_critique") or {}
    print("\n[analyze] domain:", intent.get("domain"))
    print("[analyze] terminal:", terminal, "/ attempts:", result.get("codegen_attempts"))
    print("[analyze] critic:", crit.get("verdict"), crit.get("method_issues"))
    print("\n[generated code]\n", (result.get("generated_code") or "")[:1200])
    print("\n[findings]")
    for f in result.get("key_findings", []):
        print("  -", f)
    for c in checks:
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {'' if c.passed else c.detail}")

    assert result.get("generated_code"), "generate node must produce code"
    assert result.get("codegen_attempts", 0) >= 1
    assert terminal in {"validated_result", "method_review_failed"}

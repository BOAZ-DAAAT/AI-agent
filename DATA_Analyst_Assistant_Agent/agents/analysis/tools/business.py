from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.frame import frame_from_records

@tool
def analyze_contribution(
    records: list[dict[str, Any]], dimension: str, metric: str
) -> dict[str, Any]:
    """Calculate ranked contribution, concentration, and Pareto coverage by dimension."""

    df = frame_from_records(records)
    if dimension not in df.columns or metric not in df.columns:
        raise ValueError("Dimension and metric columns are required for contribution analysis.")
    work = df[[dimension, metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    grouped = work.dropna(subset=[metric]).groupby(dimension, dropna=False)[metric].sum().sort_values(ascending=False)
    total = float(grouped.sum())
    if grouped.empty or total == 0:
        raise ValueError("Contribution analysis requires a non-zero aggregated metric.")
    share = grouped / total
    cumulative = share.cumsum()
    pareto_count = int((cumulative < 0.8).sum() + 1)
    rows = [
        {"dimension_value": str(key), "metric": float(value), "share": float(share.loc[key]), "cumulative_share": float(cumulative.loc[key])}
        for key, value in grouped.head(100).items()
    ]
    return {
        "method": "pareto_contribution",
        "dimension": dimension,
        "metric": metric,
        "group_count": len(grouped),
        "top_1_share": float(share.iloc[0]),
        "top_5_share": float(share.head(5).sum()),
        "pareto_80_group_count": pareto_count,
        "pareto_80_group_rate": float(pareto_count / len(grouped)),
        "contributions": rows,
    }

@tool
def analyze_mix_shift(
    records: list[dict[str, Any]], period_column: str, dimension: str, metric: str
) -> dict[str, Any]:
    """Explain the total metric change between the latest two periods by dimension contribution."""

    df = frame_from_records(records)
    required = [period_column, dimension, metric]
    if any(column not in df.columns for column in required):
        raise ValueError("Period, dimension, and metric columns are required for mix-shift analysis.")
    work = df[required].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[period_column, dimension, metric])
    periods = sorted(work[period_column].astype(str).unique())
    if len(periods) < 2:
        raise ValueError("Mix-shift analysis requires at least two periods.")
    previous_period, current_period = periods[-2], periods[-1]
    pivot = (
        work[work[period_column].astype(str).isin([previous_period, current_period])]
        .assign(_period=work[period_column].astype(str))
        .pivot_table(index=dimension, columns="_period", values=metric, aggfunc="sum", fill_value=0.0)
    )
    for period in (previous_period, current_period):
        if period not in pivot.columns:
            pivot[period] = 0.0
    pivot["delta"] = pivot[current_period] - pivot[previous_period]
    total_previous = float(pivot[previous_period].sum())
    total_current = float(pivot[current_period].sum())
    total_delta = total_current - total_previous
    pivot = pivot.sort_values("delta", ascending=False)
    drivers = [
        {
            "dimension_value": str(index),
            "previous": float(row[previous_period]),
            "current": float(row[current_period]),
            "delta": float(row["delta"]),
            "share_of_total_change": float(row["delta"] / total_delta) if total_delta else None,
        }
        for index, row in pivot.head(100).iterrows()
    ]
    return {
        "method": "period_contribution_change",
        "previous_period": previous_period,
        "current_period": current_period,
        "previous_total": total_previous,
        "current_total": total_current,
        "total_delta": total_delta,
        "growth_rate": float(total_delta / total_previous) if total_previous else None,
        "dimension_drivers": drivers,
    }

@tool
def simulate_scenario(
    records: list[dict[str, Any]], metric: str, change_percent: float, aggregation: str = "sum"
) -> dict[str, Any]:
    """Apply an explicit percentage change to a metric for transparent what-if analysis."""

    df = frame_from_records(records)
    if metric not in df.columns:
        raise ValueError("Metric column is required for scenario analysis.")
    values = pd.to_numeric(df[metric], errors="coerce").dropna()
    if values.empty:
        raise ValueError("Scenario analysis requires numeric metric values.")
    baseline = float(values.mean() if aggregation == "mean" else values.sum())
    projected = baseline * (1.0 + change_percent / 100.0)
    return {
        "method": "deterministic_what_if",
        "metric": metric,
        "aggregation": aggregation,
        "change_percent": change_percent,
        "baseline": baseline,
        "projected": projected,
        "absolute_change": projected - baseline,
        "assumption": "All other factors are held constant.",
    }

@tool
def optimize_business_allocation(
    records: list[dict[str, Any]],
    item_column: str,
    value_column: str,
    cost_column: str,
    budget: float,
    integer: bool = False,
    minimum_allocations: dict[str, float] | None = None,
    maximum_allocations: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Maximize linear business value under budget and per-item allocation constraints."""

    from ortools.linear_solver import pywraplp

    df = frame_from_records(records)
    required = [item_column, value_column, cost_column]
    if any(column not in df.columns for column in required):
        raise ValueError("Optimization requires item, value, and cost columns.")
    work = df[required].copy()
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work[cost_column] = pd.to_numeric(work[cost_column], errors="coerce")
    work = work.dropna()
    if work[item_column].duplicated().any():
        raise ValueError("Optimization requires one row per unique item.")
    if budget <= 0 or (work[cost_column] <= 0).any():
        raise ValueError("Budget and item costs must be positive.")
    solver_name = "CBC_MIXED_INTEGER_PROGRAMMING" if integer else "GLOP_LINEAR_PROGRAMMING"
    solver = pywraplp.Solver.CreateSolver(solver_name)
    if solver is None:
        raise RuntimeError(f"OR-Tools solver is unavailable: {solver_name}")
    minimums = minimum_allocations or {}
    maximums = maximum_allocations or {}
    variables = {}
    for _, row in work.iterrows():
        item = str(row[item_column])
        lower = float(minimums.get(item, 0.0))
        upper = float(maximums.get(item, solver.infinity()))
        variables[item] = (
            solver.IntVar(lower, upper, item) if integer else solver.NumVar(lower, upper, item)
        )
    solver.Add(
        sum(float(row[cost_column]) * variables[str(row[item_column])] for _, row in work.iterrows()) <= budget
    )
    solver.Maximize(
        sum(float(row[value_column]) * variables[str(row[item_column])] for _, row in work.iterrows())
    )
    status = solver.Solve()
    status_names = {
        pywraplp.Solver.OPTIMAL: "optimal",
        pywraplp.Solver.FEASIBLE: "feasible",
        pywraplp.Solver.INFEASIBLE: "infeasible",
        pywraplp.Solver.UNBOUNDED: "unbounded",
        pywraplp.Solver.ABNORMAL: "abnormal",
        pywraplp.Solver.NOT_SOLVED: "not_solved",
    }
    status_name = status_names.get(status, "unknown")
    if status not in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE}:
        raise ValueError(f"Optimization did not find a usable solution: {status_name}")
    allocations = {item: float(variable.solution_value()) for item, variable in variables.items()}
    budget_used = sum(
        float(row[cost_column]) * allocations[str(row[item_column])] for _, row in work.iterrows()
    )
    return {
        "method": "ortools_linear_allocation",
        "solver": solver_name,
        "status": status_name,
        "integer": integer,
        "objective_value": float(solver.Objective().Value()),
        "budget": budget,
        "budget_used": float(budget_used),
        "budget_slack": float(budget - budget_used),
        "allocations": allocations,
    }

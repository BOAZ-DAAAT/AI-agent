from __future__ import annotations

from typing import Any

import pandas as pd
from langchain_core.tools import tool
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.frame import frame_from_records

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

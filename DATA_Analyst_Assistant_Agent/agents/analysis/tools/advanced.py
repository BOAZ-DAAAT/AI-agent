from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool


def _frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


def _posterior_mean(array) -> np.ndarray:
    dimensions = [dimension for dimension in ("chain", "draw") if dimension in array.dims]
    return np.asarray(array.mean(dim=dimensions).values if dimensions else array.values)


@tool
def run_bayesian_mmm(
    records: list[dict[str, Any]],
    date_column: str,
    outcome_column: str,
    channel_columns: list[str],
    control_columns: list[str] | None = None,
    yearly_seasonality: int | None = None,
    adstock_l_max: int = 4,
    draws: int = 500,
    tune: int = 500,
    chains: int = 2,
) -> dict[str, Any]:
    """Fit a Bayesian MMM with geometric adstock and logistic saturation."""

    os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
    import arviz as az
    from pymc_marketing.mmm.multidimensional import MMM
    from pymc_marketing.mmm.components.adstock import GeometricAdstock
    from pymc_marketing.mmm.components.saturation import LogisticSaturation

    df = _frame(records)
    controls = control_columns or []
    required = [date_column, outcome_column, *channel_columns, *controls]
    if any(column not in df.columns for column in required):
        raise ValueError("MMM references columns that do not exist in the analysis data.")
    if not channel_columns:
        raise ValueError("MMM requires at least one media channel column.")
    work = df[required].copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    for column in [outcome_column, *channel_columns, *controls]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna().sort_values(date_column)
    if len(work) < 52:
        raise ValueError("MMM requires at least 52 regular time periods.")
    if work[date_column].duplicated().any():
        raise ValueError("MMM requires one row per time period after aggregation.")
    constant_channels = [column for column in channel_columns if work[column].nunique() < 2]
    if constant_channels:
        raise ValueError(f"MMM channel spend has no variation: {constant_channels}")

    model = MMM(
        date_column=date_column,
        channel_columns=channel_columns,
        target_column=outcome_column,
        control_columns=controls or None,
        adstock=GeometricAdstock(l_max=adstock_l_max),
        saturation=LogisticSaturation(),
        yearly_seasonality=yearly_seasonality,
    )
    x = work[[date_column, *channel_columns, *controls]]
    inference = model.fit(
        X=x,
        y=work[outcome_column],
        progressbar=False,
        random_seed=42,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=1,
    )
    contributions = model.compute_mean_contributions_over_time()
    channel_contributions = {
        column: float(contributions[column].sum())
        for column in channel_columns
        if column in contributions.columns
    }
    total_channel_contribution = sum(channel_contributions.values())
    shares = {
        column: value / total_channel_contribution if total_channel_contribution else 0.0
        for column, value in channel_contributions.items()
    }
    diagnostics = az.summary(inference, kind="diagnostics")
    return {
        "method": "pymc_marketing_bayesian_mmm",
        "period_count": len(work),
        "outcome_column": outcome_column,
        "channel_columns": channel_columns,
        "control_columns": controls,
        "adstock_l_max": adstock_l_max,
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "channel_contributions": channel_contributions,
        "channel_contribution_shares": shares,
        "max_r_hat": float(diagnostics["r_hat"].max()) if "r_hat" in diagnostics else None,
        "min_ess_bulk": float(diagnostics["ess_bulk"].min()) if "ess_bulk" in diagnostics else None,
    }


@tool
def estimate_probabilistic_clv(
    records: list[dict[str, Any]],
    customer_id_column: str,
    datetime_column: str,
    monetary_value_column: str,
    future_periods: int = 180,
    discount_rate: float = 0.0,
    draws: int = 500,
    tune: int = 500,
    chains: int = 2,
) -> dict[str, Any]:
    """Estimate probabilistic CLV with BG/NBD purchase and Gamma-Gamma spend models."""

    os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
    import arviz as az
    from pymc_marketing.clv import BetaGeoModel, GammaGammaModel
    from pymc_marketing.clv.utils import rfm_summary

    df = _frame(records)
    required = [customer_id_column, datetime_column, monetary_value_column]
    if any(column not in df.columns for column in required):
        raise ValueError("CLV requires customer ID, transaction time, and monetary value columns.")
    transactions = df[required].copy()
    transactions[datetime_column] = pd.to_datetime(transactions[datetime_column], errors="coerce")
    transactions[monetary_value_column] = pd.to_numeric(transactions[monetary_value_column], errors="coerce")
    transactions = transactions.dropna()
    if transactions[customer_id_column].nunique() < 20:
        raise ValueError("Probabilistic CLV requires at least twenty customers.")
    summary = rfm_summary(
        transactions,
        customer_id_col=customer_id_column,
        datetime_col=datetime_column,
        monetary_value_col=monetary_value_column,
        time_unit="D",
    )
    repeat = summary[(summary["frequency"] > 0) & (summary["monetary_value"] > 0)].copy()
    if len(repeat) < 10:
        raise ValueError("Gamma-Gamma CLV requires at least ten repeat customers with positive spend.")

    fit_kwargs = {
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "cores": 1,
        "progressbar": False,
        "random_seed": 42,
    }
    transaction_model = BetaGeoModel(data=summary)
    bg_inference = transaction_model.fit(method="mcmc", **fit_kwargs)
    spend_model = GammaGammaModel(data=repeat)
    gg_inference = spend_model.fit(method="mcmc", **fit_kwargs)

    probability_alive = transaction_model.expected_probability_alive(summary)
    expected_purchases = transaction_model.expected_purchases(summary, future_t=future_periods)
    expected_spend = spend_model.expected_customer_spend(repeat)
    clv = spend_model.expected_customer_lifetime_value(
        transaction_model,
        repeat,
        future_t=future_periods,
        discount_rate=discount_rate,
        time_unit="D",
    )
    alive_mean = _posterior_mean(probability_alive).reshape(-1)
    purchase_mean = _posterior_mean(expected_purchases).reshape(-1)
    spend_mean = _posterior_mean(expected_spend).reshape(-1)
    clv_mean = _posterior_mean(clv).reshape(-1)
    customers = summary["customer_id"].astype(str).tolist()
    repeat_customers = repeat["customer_id"].astype(str).tolist()
    top_clv = sorted(
        [
            {
                "customer_id": customer,
                "expected_clv": float(value),
                "expected_spend": float(spend_mean[index]),
            }
            for index, (customer, value) in enumerate(zip(repeat_customers, clv_mean))
        ],
        key=lambda item: item["expected_clv"],
        reverse=True,
    )[:100]
    bg_diagnostics = az.summary(bg_inference, kind="diagnostics")
    gg_diagnostics = az.summary(gg_inference, kind="diagnostics")
    return {
        "method": "pymc_marketing_bg_nbd_gamma_gamma",
        "customer_count": len(summary),
        "repeat_customer_count": len(repeat),
        "future_periods": future_periods,
        "discount_rate": discount_rate,
        "mean_probability_alive": float(np.mean(alive_mean)),
        "mean_expected_purchases": float(np.mean(purchase_mean)),
        "mean_expected_clv": float(np.mean(clv_mean)),
        "top_customer_clv": top_clv,
        "max_r_hat": float(max(bg_diagnostics["r_hat"].max(), gg_diagnostics["r_hat"].max())),
        "customers": customers[:100],
    }


@tool
def analyze_geospatial_hotspots(
    records: list[dict[str, Any]],
    latitude_column: str,
    longitude_column: str,
    metric: str,
    k_neighbors: int = 5,
    permutations: int = 999,
) -> dict[str, Any]:
    """Measure global and local spatial autocorrelation for point-based business metrics."""

    import geopandas as gpd
    from esda import Moran, Moran_Local
    from libpysal.weights import KNN

    df = _frame(records)
    required = [latitude_column, longitude_column, metric]
    if any(column not in df.columns for column in required):
        raise ValueError("Spatial analysis requires latitude, longitude, and metric columns.")
    work = df[required].copy()
    for column in required:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna()
    work = work[
        work[latitude_column].between(-90, 90)
        & work[longitude_column].between(-180, 180)
    ]
    if len(work) < max(10, k_neighbors + 2):
        raise ValueError("Spatial autocorrelation requires at least ten valid points.")
    points = gpd.GeoDataFrame(
        work,
        geometry=gpd.points_from_xy(work[longitude_column], work[latitude_column]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")
    weights = KNN.from_dataframe(points, k=min(k_neighbors, len(points) - 1))
    weights.transform = "R"
    values = points[metric].to_numpy(dtype=float)
    np.random.seed(42)
    global_moran = Moran(values, weights, permutations=permutations)
    local_moran = Moran_Local(
        values,
        weights,
        permutations=permutations,
        seed=42,
        alternative="two-sided",
    )
    significant = local_moran.p_sim < 0.05
    hotspot_positions = np.flatnonzero(significant & (local_moran.q == 1)).tolist()
    coldspot_positions = np.flatnonzero(significant & (local_moran.q == 3)).tolist()
    return {
        "method": "moran_spatial_autocorrelation",
        "point_count": len(points),
        "source_crs": "EPSG:4326",
        "analysis_crs": "EPSG:3857",
        "k_neighbors": weights.k,
        "global_moran_i": float(global_moran.I),
        "global_p_value_simulated": float(global_moran.p_sim),
        "hotspot_count": len(hotspot_positions),
        "coldspot_count": len(coldspot_positions),
        "hotspot_row_positions": hotspot_positions[:100],
        "coldspot_row_positions": coldspot_positions[:100],
        "permutations": permutations,
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

    df = _frame(records)
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

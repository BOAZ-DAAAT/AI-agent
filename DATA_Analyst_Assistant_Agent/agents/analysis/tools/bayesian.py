from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.frame import frame_from_records
import os

from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.bayesian import posterior_mean
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.datetime import safe_datetime_series

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

    df = frame_from_records(records)
    controls = control_columns or []
    required = [date_column, outcome_column, *channel_columns, *controls]
    if any(column not in df.columns for column in required):
        raise ValueError("MMM references columns that do not exist in the analysis data.")
    if not channel_columns:
        raise ValueError("MMM requires at least one media channel column.")
    work = df[required].copy()
    work[date_column] = safe_datetime_series(work[date_column])
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

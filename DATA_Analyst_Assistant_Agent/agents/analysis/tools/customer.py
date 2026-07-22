from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.frame import frame_from_records
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.bayesian import posterior_mean
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.datetime import safe_datetime_series

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

    df = frame_from_records(records)
    required = [customer_id_column, datetime_column, monetary_value_column]
    if any(column not in df.columns for column in required):
        raise ValueError("CLV requires customer ID, transaction time, and monetary value columns.")
    transactions = df[required].copy()
    transactions[datetime_column] = safe_datetime_series(transactions[datetime_column])
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
    alive_mean = posterior_mean(probability_alive).reshape(-1)
    purchase_mean = posterior_mean(expected_purchases).reshape(-1)
    spend_mean = posterior_mean(expected_spend).reshape(-1)
    clv_mean = posterior_mean(clv).reshape(-1)
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

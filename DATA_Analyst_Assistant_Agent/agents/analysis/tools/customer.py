from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.frame import frame_from_records
import os
from collections import Counter

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.bayesian import posterior_mean

@tool
def analyze_cohort_retention(
    records: list[dict[str, Any]],
    entity_id_column: str,
    event_time_column: str,
    cohort_time_column: str | None = None,
    period_unit: str = "month",
) -> dict[str, Any]:
    """Build acquisition cohorts and calculate period-by-period entity retention."""

    df = frame_from_records(records)
    required = [entity_id_column, event_time_column]
    if any(column not in df.columns for column in required):
        raise ValueError("Entity ID and event time columns are required for cohort analysis.")
    work = df[required + ([cohort_time_column] if cohort_time_column else [])].copy()
    work[event_time_column] = pd.to_datetime(work[event_time_column], errors="coerce")
    work = work.dropna(subset=[entity_id_column, event_time_column])
    if cohort_time_column:
        work[cohort_time_column] = pd.to_datetime(work[cohort_time_column], errors="coerce")
        work["_cohort_time"] = work[cohort_time_column]
    else:
        work["_cohort_time"] = work.groupby(entity_id_column)[event_time_column].transform("min")
    if period_unit == "week":
        work["_cohort"] = work["_cohort_time"].dt.to_period("W").astype(str)
        work["_period_index"] = ((work[event_time_column] - work["_cohort_time"]).dt.days // 7).astype(int)
    else:
        work["_cohort"] = work["_cohort_time"].dt.to_period("M").astype(str)
        work["_period_index"] = (
            (work[event_time_column].dt.year - work["_cohort_time"].dt.year) * 12
            + work[event_time_column].dt.month - work["_cohort_time"].dt.month
        ).astype(int)
    work = work[work["_period_index"] >= 0]
    active = work.groupby(["_cohort", "_period_index"])[entity_id_column].nunique()
    cohort_sizes = active.groupby(level=0).first()
    matrix: list[dict[str, Any]] = []
    for (cohort, period), count in active.items():
        size = int(cohort_sizes.loc[cohort])
        matrix.append({
            "cohort": str(cohort),
            "period_index": int(period),
            "active_entities": int(count),
            "cohort_size": size,
            "retention_rate": float(count / size) if size else 0.0,
        })
    return {
        "method": "cohort_retention",
        "entity_id_column": entity_id_column,
        "event_time_column": event_time_column,
        "period_unit": period_unit,
        "cohort_count": int(len(cohort_sizes)),
        "entity_count": int(work[entity_id_column].nunique()),
        "retention_matrix": matrix,
    }

@tool
def analyze_funnel(
    records: list[dict[str, Any]],
    entity_id_column: str,
    event_column: str,
    event_time_column: str,
    steps: list[str],
) -> dict[str, Any]:
    """Measure strict ordered funnel conversion and drop-off by entity."""

    df = frame_from_records(records)
    if not steps or len(steps) < 2:
        raise ValueError("Funnel analysis requires at least two ordered steps.")
    required = [entity_id_column, event_column, event_time_column]
    if any(column not in df.columns for column in required):
        raise ValueError("Entity, event, and event-time columns are required for funnel analysis.")
    work = df[required].copy()
    work[event_time_column] = pd.to_datetime(work[event_time_column], errors="coerce")
    work = work.dropna().sort_values([entity_id_column, event_time_column])
    reached = [0] * len(steps)
    for _, group in work.groupby(entity_id_column):
        next_step = 0
        for event in group[event_column].astype(str):
            if next_step < len(steps) and event == steps[next_step]:
                reached[next_step] += 1
                next_step += 1
    first = reached[0]
    result_steps = []
    for index, (step, count) in enumerate(zip(steps, reached)):
        previous = reached[index - 1] if index > 0 else count
        result_steps.append({
            "step": step,
            "entities": int(count),
            "conversion_from_start": float(count / first) if first else 0.0,
            "conversion_from_previous": float(count / previous) if previous else 0.0,
            "dropoff_from_previous": int(previous - count) if index > 0 else 0,
        })
    return {
        "method": "strict_ordered_funnel",
        "entity_count": int(work[entity_id_column].nunique()),
        "steps": result_steps,
        "overall_conversion_rate": float(reached[-1] / first) if first else 0.0,
    }

@tool
def analyze_journey(
    records: list[dict[str, Any]],
    entity_id_column: str,
    event_column: str,
    event_time_column: str,
    max_steps: int = 10,
    top_n: int = 20,
) -> dict[str, Any]:
    """Summarize common ordered entity paths and event-to-event transitions."""

    df = frame_from_records(records)
    required = [entity_id_column, event_column, event_time_column]
    if any(column not in df.columns for column in required):
        raise ValueError("Entity, event, and event-time columns are required for journey analysis.")
    work = df[required].copy()
    work[event_time_column] = pd.to_datetime(work[event_time_column], errors="coerce")
    work = work.dropna().sort_values([entity_id_column, event_time_column])
    paths: Counter[tuple[str, ...]] = Counter()
    transitions: Counter[tuple[str, str]] = Counter()
    for _, group in work.groupby(entity_id_column):
        events = group[event_column].astype(str).tolist()[:max_steps]
        if events:
            paths[tuple(events)] += 1
        transitions.update(zip(events, events[1:]))
    return {
        "method": "ordered_journey_paths",
        "entity_count": int(work[entity_id_column].nunique()),
        "top_paths": [
            {"path": list(path), "entities": count}
            for path, count in paths.most_common(top_n)
        ],
        "top_transitions": [
            {"from_event": pair[0], "to_event": pair[1], "count": count}
            for pair, count in transitions.most_common(top_n)
        ],
    }

@tool
def segment_entities(
    records: list[dict[str, Any]], entity_id_column: str, features: list[str], n_clusters: int = 4
) -> dict[str, Any]:
    """Cluster entities using standardized numeric behavior features."""

    df = frame_from_records(records)
    selected = [column for column in features if column in df.columns and pd.api.types.is_numeric_dtype(df[column])]
    if entity_id_column not in df.columns or len(selected) < 2:
        raise ValueError("Segmentation requires an entity ID and at least two numeric features.")
    entity_frame = df.groupby(entity_id_column, as_index=False)[selected].mean()
    if len(entity_frame) <= n_clusters:
        raise ValueError("Entity count must be larger than n_clusters.")
    values = SimpleImputer(strategy="median").fit_transform(entity_frame[selected])
    values = StandardScaler().fit_transform(values)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = model.fit_predict(values)
    entity_frame["segment"] = labels
    profiles = entity_frame.groupby("segment")[selected].mean().reset_index().to_dict(orient="records")
    sizes = entity_frame["segment"].value_counts().sort_index().to_dict()
    return {
        "method": "kmeans_segmentation",
        "entity_id_column": entity_id_column,
        "features": selected,
        "n_clusters": n_clusters,
        "entity_count": len(entity_frame),
        "silhouette_score": float(silhouette_score(values, labels)),
        "segment_sizes": {str(key): int(value) for key, value in sizes.items()},
        "segment_profiles": profiles,
    }

@tool
def analyze_rfm(
    records: list[dict[str, Any]], customer_id_column: str, event_time_column: str, amount_column: str
) -> dict[str, Any]:
    """Calculate recency, frequency, monetary scores and customer segment counts."""

    df = frame_from_records(records)
    required = [customer_id_column, event_time_column, amount_column]
    if any(column not in df.columns for column in required):
        raise ValueError("Customer ID, event time, and amount columns are required for RFM analysis.")
    work = df[required].copy()
    work[event_time_column] = pd.to_datetime(work[event_time_column], errors="coerce")
    work[amount_column] = pd.to_numeric(work[amount_column], errors="coerce")
    work = work.dropna()
    reference = work[event_time_column].max() + pd.Timedelta(days=1)
    rfm = work.groupby(customer_id_column).agg(
        recency=(event_time_column, lambda value: int((reference - value.max()).days)),
        frequency=(event_time_column, "count"),
        monetary=(amount_column, "sum"),
    )
    if len(rfm) < 5:
        raise ValueError("RFM analysis requires at least five customers.")
    rfm["r_score"] = pd.qcut(rfm["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["score"] = rfm[["r_score", "f_score", "m_score"]].sum(axis=1)
    rfm["segment"] = pd.cut(
        rfm["score"], bins=[0, 6, 9, 12, 15], labels=["at_risk", "regular", "loyal", "champion"]
    )
    return {
        "method": "rfm_quintile_scoring",
        "customer_count": len(rfm),
        "reference_date": str(reference.date()),
        "segment_counts": {str(key): int(value) for key, value in rfm["segment"].value_counts().items()},
        "summary": rfm[["recency", "frequency", "monetary", "score"]].describe().to_dict(),
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

    df = frame_from_records(records)
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

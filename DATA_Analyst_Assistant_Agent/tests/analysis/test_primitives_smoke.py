"""Smoke tests for the vetted heavy tools kept as codegen primitives.

These 5 tools are NOT reimplemented by generated code; they are injected into the
sandbox (see nodes/generate._PRIMITIVE_TOOLS) and called as vetted primitives.
The smoke tests guard that their public .invoke() contract keeps working.
"""

from __future__ import annotations

import pytest

from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.generate import _PRIMITIVE_TOOLS
from DATA_Analyst_Assistant_Agent.agents.analysis.tools import ANALYSIS_TOOLS


def test_primitive_tools_are_registered() -> None:
    for name in _PRIMITIVE_TOOLS:
        assert name in ANALYSIS_TOOLS, f"primitive {name} missing from ANALYSIS_TOOLS"


def test_survival_primitive_runs() -> None:
    records = [
        {"duration": float(i + 1), "observed": 1 if i % 3 else 0} for i in range(10)
    ]
    out = ANALYSIS_TOOLS["analyze_survival"].invoke({
        "records": records,
        "duration_column": "duration",
        "event_observed_column": "observed",
    })
    assert out["observation_count"] == 10


def test_geospatial_primitive_runs() -> None:
    pytest.importorskip("geopandas")
    pytest.importorskip("esda")
    pytest.importorskip("libpysal")

    records = [
        {"latitude": 37.50 + i * 0.001, "longitude": 127.00 + i * 0.001, "sales": 100.0 if i < 6 else 10.0}
        for i in range(12)
    ]
    out = ANALYSIS_TOOLS["analyze_geospatial_hotspots"].invoke({
        "records": records,
        "latitude_column": "latitude",
        "longitude_column": "longitude",
        "metric": "sales",
        "k_neighbors": 3,
        "permutations": 19,
    })
    assert out["point_count"] == 12
    assert "global_moran_i" in out


def test_optimization_primitive_runs() -> None:
    pytest.importorskip("ortools")

    out = ANALYSIS_TOOLS["optimize_business_allocation"].invoke({
        "records": [
            {"channel": "search", "value": 12.0, "cost": 4.0},
            {"channel": "social", "value": 8.0, "cost": 5.0},
        ],
        "item_column": "channel",
        "value_column": "value",
        "cost_column": "cost",
        "budget": 100.0,
        "maximum_allocations": {"search": 15.0, "social": 10.0},
    })
    assert out["status"] == "optimal"
    assert out["budget_used"] <= 100.0


def test_mmm_and_clv_primitives_enforce_data_contracts() -> None:
    pytest.importorskip("arviz")
    pytest.importorskip("pymc_marketing")

    # Full Bayesian fits are heavy; assert the vetted data-contract guards hold.
    with pytest.raises(ValueError, match="at least 52"):
        ANALYSIS_TOOLS["run_bayesian_mmm"].invoke({
            "records": [
                {"date": "2024-01-01", "sales": 10.0, "search_spend": 2.0},
                {"date": "2024-01-08", "sales": 12.0, "search_spend": 3.0},
            ],
            "date_column": "date",
            "outcome_column": "sales",
            "channel_columns": ["search_spend"],
            "draws": 5, "tune": 5, "chains": 1,
        })
    with pytest.raises(ValueError, match="at least twenty"):
        ANALYSIS_TOOLS["estimate_probabilistic_clv"].invoke({
            "records": [
                {"customer": "A", "date": "2024-01-01", "amount": 10.0},
                {"customer": "A", "date": "2024-02-01", "amount": 12.0},
            ],
            "customer_id_column": "customer",
            "datetime_column": "date",
            "monetary_value_column": "amount",
            "draws": 5, "tune": 5, "chains": 1,
        })

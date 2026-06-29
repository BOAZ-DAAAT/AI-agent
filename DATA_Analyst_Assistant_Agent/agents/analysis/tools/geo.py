from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.frame import frame_from_records

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

    df = frame_from_records(records)
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

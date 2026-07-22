from __future__ import annotations

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize


def _seller_delivery_frame() -> pd.DataFrame:
    n = 10_054
    return pd.DataFrame(
        {
            "seller_id": [f"seller_{i % 1581:04d}" for i in range(n)],
            "order_id": [f"order_{i:05d}" for i in range(n)],
            "order_purchase_month": [f"2018-{(i % 12) + 1:02d}-01" for i in range(n)],
            "order_purchase_timestamp": [f"2018-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}T08:00:00" for i in range(n)],
            "order_delivered_customer_date": [f"2018-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}T18:00:00" for i in range(n)],
            "delivery_days": [(i % 30) + 1 for i in range(n)],
            "latest_review_score": [(i % 5) + 1 for i in range(n)],
        }
    )


def test_crosstab_heatmap_filters_only_needed_columns_for_high_card_key(tmp_path, monkeypatch):
    monkeypatch.setattr(visualize, "OUTPUT_DIR", str(tmp_path))
    df = _seller_delivery_frame()

    result = visualize.plot_crosstab_heatmap(df, cat_a="seller_id")

    assert result["chart_paths"]
    assert result["stats"]["shape"] == [15, 12]
    assert (tmp_path / "crosstab_seller_id_x_order_purchase_month.png").exists()


def test_cluster_scatter_annotations_filter_only_needed_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(visualize, "OUTPUT_DIR", str(tmp_path))
    df = _seller_delivery_frame().head(60).copy()
    df["cluster"] = [i % 2 for i in range(len(df))]

    result = visualize.plot_cluster_scatter(
        df,
        x_col="delivery_days",
        y_col="latest_review_score",
        cluster_col="cluster",
        key_col="seller_id",
    )

    assert result["chart_paths"]
    assert (tmp_path / "cluster_scatter_delivery_days_vs_latest_review_score.png").exists()

from __future__ import annotations

from datetime import date, datetime

import pandas as pd


def safe_datetime_series(series: pd.Series) -> pd.Series:
    """Parse datetime-like values scalar-by-scalar.

    Some Windows/Pandas combinations can hard-crash on vectorized
    ``pd.to_datetime(series)`` for object date columns. Returning Python
    ``datetime`` objects avoids datetime64 block operations in downstream code.
    """
    parsed_values: list[datetime | None] = []
    for value in series:
        if value is None:
            parsed_values.append(None)
            continue
        if isinstance(value, pd.Timestamp):
            parsed_values.append(value.to_pydatetime())
            continue
        if isinstance(value, datetime):
            parsed_values.append(value)
            continue
        if isinstance(value, date):
            parsed_values.append(datetime.combine(value, datetime.min.time()))
            continue
        text = str(value).strip()
        if not text or text.lower() in {"nat", "nan", "none"}:
            parsed_values.append(None)
            continue
        try:
            parsed = pd.to_datetime(text, errors="coerce")
            parsed_values.append(None if pd.isna(parsed) else parsed.to_pydatetime())
        except Exception:  # noqa: BLE001
            parsed_values.append(None)
    return pd.Series(parsed_values, index=series.index, dtype="object")

from __future__ import annotations

from typing import Any

import pandas as pd


def frame_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)

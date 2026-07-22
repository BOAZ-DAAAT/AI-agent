from __future__ import annotations

from packaging.version import Version

import numpy as np
import pandas as pd


def test_analysis_runtime_uses_locked_pandas_stack() -> None:
    assert Version(pd.__version__) == Version("2.3.3")
    assert Version(np.__version__) == Version("2.4.6")


def test_daily_datetime_ranges_do_not_crash_analysis_runtime() -> None:
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    timedeltas = pd.timedelta_range(start="1 day", periods=3, freq="D")

    assert len(dates) == 30
    assert str(dates[0]) == "2026-01-01 00:00:00"
    assert len(timedeltas) == 3

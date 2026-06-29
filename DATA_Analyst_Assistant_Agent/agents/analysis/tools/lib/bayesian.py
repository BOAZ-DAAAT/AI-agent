from __future__ import annotations

import numpy as np


def posterior_mean(array) -> np.ndarray:
    dimensions = [dimension for dimension in ("chain", "draw") if dimension in array.dims]
    return np.asarray(array.mean(dim=dimensions).values if dimensions else array.values)

"""Population Stability Index."""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi_from_labels(expected: pd.Series, actual: pd.Series, eps: float = 1e-4) -> float:
    cats = sorted(set(expected.unique()) | set(actual.unique()), key=str)
    e = expected.value_counts(normalize=True).reindex(cats).fillna(0) + eps
    a = actual.value_counts(normalize=True).reindex(cats).fillna(0) + eps
    return float(((a - e) * np.log(a / e)).sum())


def psi_numeric(expected, actual, n: int = 10) -> float:
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n + 1)[1:-1]))
    return psi_from_labels(pd.Series(np.searchsorted(edges, expected, "left")),
                           pd.Series(np.searchsorted(edges, actual, "left")))


def psi_status(v: float) -> str:
    return "stable" if v < 0.1 else ("moderate shift" if v < 0.25 else "significant shift")
